"""Six Round 4 cases; common-input attention and cache-fork interventions."""
from collections import defaultdict
import argparse
from pathlib import Path
import json

import torch
import transformers

from gqa_probe.fixed_k_v import (candidate_qk,gather_tokens,keep_mask,masked_output,
                                probabilities,ranked_pages)
from gqa_probe.oracle_attention import OracleAttention
from gqa_probe.page_read_budget import page_history
from gqa_probe.paired_statistics import score_prediction
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.ruler_experiment import _write_json
from gqa_probe.ruler_inference import _copy_prefix,generate_from_prefill
from gqa_probe.vskip_attention import FixedKAttention,VPolicy
from gqa_probe.vskip_summary import StaticOnline,StaticPrefix,weights
from scripts.run_ruler_round3 import load_rows


ROOT=Path("results/round5_shared_refill")
OLD=Path("results/round4_fixed_k_v/data/test")
MODEL=Path("models/Qwen3-4B-Instruct-2507")
CASES=("0013","0017","0065","0069","0096","0072")
POLICIES={"all":VPolicy("q6"), "drop":VPolicy("q6",mode="mass",threshold=.05),
          "refill":VPolicy("q6",mode="mass",threshold=.05,refill=True)}


def metadata(cache):
    return tuple(StaticPrefix.build(key,32,("q6",)) for key in cache.key_cache)


def step(model,device,cache,token):
    return model(torch.tensor([[token]],device=device),past_key_values=cache,
                 use_cache=True,logits_to_keep=1).logits[0,-1]


class SameState(OracleAttention):
    """All three Round 4 rules see the anchor's exact Q/K/V at each layer."""
    def __init__(self,prefix,policy,evidence,all_evidence):
        super().__init__()
        self.prefix=prefix
        self.mode=policy
        self.evidence=set(evidence)
        self.all_evidence=set(all_evidence)
        self.rows=[]
        self.plans={}
        self.swaps=[]
        self.online={}
        self.refiller=FixedKAttention(POLICIES["refill"])

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        layer=module.layer_idx
        h=page_history(key.shape[-2],32)
        heads=query.shape[1]//key.shape[1]
        q=query.reshape(-1,heads,query.shape[-1]).float()
        scale=kwargs.get("scaling") or q.shape[-1]**-.5
        if layer not in self.online:
            self.online[layer]=StaticOnline(self.prefix[layer],("q6",))
        rec=self.online[layer].append(key[0,:,-1],h.tokens)["q6"]
        ranks=ranked_pages(weights(q,rec,"q6",scale),h)
        count=POLICIES["all"].count(h.tokens)
        ids,order=ranks[:,:count].sort(dim=-1)
        logits,pos,valid,force=candidate_qk(q,key,ids,h,scale)
        p=probabilities(logits)
        full=keep_mask(logits,p,valid,force,"all",0)
        drop=keep_mask(logits,p,valid,force,"mass",.05)
        full_path=(ids,logits,pos,valid,full)
        drop_path=(ids,logits,pos,valid,drop)
        add_path=self.refiller.refill(h,q,key,scale,ranks,ids,logits,pos,valid,drop)
        rg=ranks.cpu().tolist()
        for name,(chosen,z,positions,available,pv) in zip(("all","drop","refill"),(full_path,drop_path,add_path)):
            live=available.any(-1).cpu().tolist()
            pages=chosen.cpu().tolist()
            retained=pv.any((-3,-1)).cpu().tolist()
            mass=probabilities(z).sum(-1).cpu().tolist()
            for g,gs in enumerate(pages):
                for evidence in self.evidence:
                    slot=next((i for i,page in enumerate(gs) if page==evidence and live[g][i]),None)
                    if slot is not None:
                        self.rows.append(dict(policy=name,layer=layer,group=g,page=evidence,
                                              stage="initial" if slot<count else "added",
                                              v_retained=retained[g][slot],
                                              page_mass_by_head=[m[slot] for m in mass[g]]))
            if name!="refill":
                continue
            replacement=chosen.clone()
            for g,gs in enumerate(pages):
                slots=[i for i,pid in enumerate(gs) if i>=count and pid in self.evidence and
                       live[g][i] and retained[g][i]]
                if not slots:
                    continue
                i=max(slots,key=lambda s:max(head[s] for head in mass[g]))
                unavailable={pid for j,pid in enumerate(gs) if live[g][j]}|self.all_evidence
                new=next((j for j in rg[g] if j not in unavailable),None)
                if new is None:
                    continue
                replacement[g,i]=new
                self.swaps.append(dict(layer=layer,group=g,dropped=gs[i],replacement=new,
                                       page_mass_by_head=[m[i] for m in mass[g]]))
            self.plans[layer]=(chosen,replacement,available,pv)
        paths={"all":full_path,"drop":drop_path,"refill":add_path}
        _,z,po,_,pv=paths[self.mode]
        out=masked_output(z,gather_tokens(value,po),pv)
        return out.reshape(1,query.shape[1],1,q.shape[-1]).to(query.dtype).transpose(1,2),None


class Intervention(FixedKAttention):
    """One equal-page K/V plan swap followed by the ordinary refill path."""
    def __init__(self,prefix,plans,replace):
        super().__init__(POLICIES["refill"],prefix=prefix)
        self.plans=plans
        self.replace=replace

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step!=0:
            return super().attention(module,query,key,value,attention_mask,**kwargs)
        layer=module.layer_idx
        h=page_history(key.shape[-2],32)
        chosen,new,live,pv=self.plans[layer]
        # Advance online summary even while the comparison step uses fixed IDs.
        self.online[layer]=StaticOnline(self.prefix[layer],("q6",))
        self.online[layer].append(key[0,:,-1],h.tokens)
        dim=query.shape[-1]
        q=query.reshape(key.shape[1],-1,dim).float()
        ids=new if self.replace else chosen
        z,po,valid,_=candidate_qk(q,key,ids,h,kwargs.get("scaling") or dim**-.5,live.any(-1))
        torch._assert_async((valid==live).all(),"intervention changed physical page lengths")
        act=masked_output(z,gather_tokens(value,po),pv)
        return act.reshape(1,query.shape[1],1,dim).to(query.dtype).transpose(1,2),None


def onset_and_end(tokenizer,tokens,reference):
    text=tokenizer.decode(tokens,skip_special_tokens=True)
    hit=text.lower().find(reference.lower())
    if hit<0:
        return []
    lengths=[len(tokenizer.decode(tokens[:i],skip_special_tokens=True)) for i in range(1,len(tokens)+1)]
    return [next(i for i,n in enumerate(lengths) if n>hit),
            next(i for i,n in enumerate(lengths) if n>=hit+len(reference))]


def continuation(model,device,cache,prefix,anchor,at,first,pol,eos,limit):
    produced=list(anchor[:at])
    token=first
    with pol:
        while len(produced)<limit and token not in eos:
            pol.step=len(produced)-at
            produced.append(token)
            token=int(step(model,device,cache,token).argmax())
    return produced


@torch.inference_mode()
def run_case(model,tokenizer,eos,row):
    state=prefill_for_quantization(model,row["input_ids"],"affine32")
    device=state.last_logits.device
    pref=metadata(state.cache)
    seqs={k:generate_from_prefill(model,state,row["max_new_tokens"],eos,FixedKAttention(policy,prefix=pref)).token_ids
          for k,policy in POLICIES.items()}
    text={k:tokenizer.decode(t,skip_special_tokens=True) for k,t in seqs.items()}
    scores={k:score_prediction(t,row["references"]) for k,t in text.items()}
    subject=row["sample_id"].rsplit("/",1)[-1]
    anchor_mode="drop" if subject in ("0017","0072") else "refill"
    weak="refill" if subject=="0072" else ("all" if subject=="0017" else "drop")
    anchor=seqs[anchor_mode]
    target=[r for r in row["references"] if r.lower() in text[anchor_mode].lower() and r.lower() not in text[weak].lower()]
    if len(target)!=1:
        raise ValueError(f"expected one missing reference {subject}: {scores}")
    target=target[0]
    all_spans={i//32 for a,b in row["gold_spans"] for i in range(a,b)}
    evidence=set()
    for x in row["source_metadata"]["gold_evidence"]:
        if x["value"]==target:
            a,b=x["association_span"]
            evidence.update(range(a//32,(b-1)//32+1))
    diff=next(i for i,ts in enumerate(zip(*seqs.values())) if len(set(ts))>1)
    onset,end=onset_and_end(tokenizer,anchor,target)
    events=defaultdict(list)
    for at,kind in ((diff,"first_divergence"),(onset,"missing_value_onset"),(end,"missing_value_complete")):
        if at:
            events[at].append(kind)
    cache=_copy_prefix(state.cache)
    del state
    ledger=[]
    online=FixedKAttention(POLICIES[anchor_mode],prefix=pref)
    for step_id,token in enumerate(anchor[:max(events)]):
        at=step_id+1
        if at in events:
            snapshot=metadata(cache)
            observer=SameState(snapshot,anchor_mode,evidence,all_spans)
            observer.step=0
            with observer:
                reference=step(model,device,_copy_prefix(cache),token)
            fork={}
            for name,policy in POLICIES.items():
                gate=FixedKAttention(policy,prefix=snapshot)
                gate.step=0
                with gate:
                    logit=step(model,device,_copy_prefix(cache),token)
                fork[name]=dict(correct_continuation_logit=logit[anchor[at]].item(),
                                margin_to_argmax=(logit[anchor[at]]-logit.max()).item(),
                                is_correct_continuation_argmax=bool(logit.argmax()==anchor[at]))
            record=dict(at=at,events=events[at],correct_token_id=anchor[at],
                        cache_forks=fork,common_qkv_evidence=observer.rows)
            reference=None
            if at==onset and observer.swaps:
                for replace in (False,True):
                    fork_cache=_copy_prefix(cache)
                    policy=Intervention(snapshot,observer.plans,replace)
                    output=continuation(model,device,fork_cache,snapshot,anchor,at-1,token,
                                        policy,eos,row["max_new_tokens"])
                    rendered=tokenizer.decode(output,skip_special_tokens=True)
                    key="replacement" if replace else "control"
                    record.setdefault("intervention",{"swaps":observer.swaps})[key]=dict(
                        **score_prediction(rendered,row["references"]),
                        target_hit=target.lower() in rendered.lower())
                    del fork_cache
            ledger.append(record)
        online.step=step_id
        with online:
            step(model,device,cache,token)
    return dict(sample_id=row["sample_id"],anchor=anchor_mode,reference_hit_branch=scores,
                evidence_pages=sorted(evidence),events=ledger)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"mechanism")
    parser.add_argument("--cases",nargs="+",choices=CASES,default=CASES)
    args=parser.parse_args()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    _,_,rows=load_rows(OLD,"test")
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",
        low_cpu_mem_usage=True,local_files_only=True).eval()
    tokenizer=transformers.AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    for tag in args.cases:
        file=output/f"{tag}.json"
        if not file.exists():
            row=next(r for r in rows if r["sample_id"].endswith(tag))
            _write_json(file,run_case(model,tokenizer,eos,row))
        print("mechanism",tag,flush=True)


if __name__=="__main__":
    main()
