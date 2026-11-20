"""One frozen two-bucket centroid diagnostic on the 0006 teacher-forced states."""
import json
from pathlib import Path

import torch
import transformers

from gqa_probe.fixed_k_v import ranked_pages
from gqa_probe.page_read_budget import page_history
from gqa_probe.quantization_round2 import prefill_for_quantization
from gqa_probe.ruler_experiment import _write_json
from gqa_probe.ruler_inference import _copy_prefix, generate_from_prefill
from gqa_probe.vskip_attention import FixedKAttention, VPolicy
from gqa_probe.vskip_summary import StaticPrefix, quantize_vector
from scripts.run_round4_fixed_k_v import MODEL, ORIGINAL, ROOT
from scripts.run_ruler_round3 import load_rows


def two_buckets(key, q, scaling):
    g, tokens, d = key.shape
    size = 32
    z = torch.nn.functional.pad(key.float(), (0,0,0,(-tokens)%size)).unflatten(1, (-1,size))
    valid = (torch.arange(z.shape[1]*size,device=key.device)<tokens).reshape(1,-1,size)
    side = z[...,0] >= 0
    terms=[]
    for active in (side & valid, ~side & valid):
        n=active.sum(-1,keepdim=True)
        center=(z*active[...,None]).sum(2)/n.clamp_min(1)
        code, step=quantize_vector(center,8,0)
        decoded=code.float()*step
        terms.append(torch.bmm(q,decoded.transpose(1,2))*scaling+n.transpose(1,2).float().log())
    return torch.logaddexp(*terms).softmax(-1).sum(1)


class Probe(FixedKAttention):
    def __init__(self,prefix,targets,gold):
        super().__init__(VPolicy("q8",legacy_layout=True),prefix=prefix)
        self.targets=targets
        self.gold=gold
        self.rows=[]

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step in self.targets:
            q=query.reshape(key.shape[1],-1,query.shape[-1]).float()
            scaling=kwargs.get("scaling") or q.shape[-1]**-.5
            h=page_history(key.shape[-2],32)
            ranks=(ranked_pages(two_buckets(key[0],q,scaling),h).argsort(-1)[:,self.gold]+1).tolist()
            for group,gg in enumerate(ranks):
                for page,rank in zip(self.gold,gg):
                    self.rows.append(dict(decode_step=self.step,layer=module.layer_idx,group=group,page=page,
                                          dual_rank=rank,dual_budget=self.policy.count(h.tokens)))
        return super().attention(module,query,key,value,attention_mask,**kwargs)


@torch.inference_mode()
def main():
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(6)
    _,_,rows=load_rows(ORIGINAL,"calibration")
    datum=next(r for r in rows if r["task"]=="niah_multivalue" and r["sample_id"].endswith("0006"))
    diag=json.loads((ROOT/"report/diagnostic/0006_diagnostic.json").read_text())
    targets=sorted({r["decode_step"] for r in diag["rows"]})
    model=transformers.AutoModelForCausalLM.from_pretrained(
        MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa",device_map="cuda",
        low_cpu_mem_usage=True,local_files_only=True).eval()
    state=prefill_for_quantization(model,datum["input_ids"],"affine32")
    prefix=tuple(StaticPrefix.build(k,32,("q8",)) for k in state.cache.key_cache)
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    base=generate_from_prefill(model,state,datum["max_new_tokens"],eos,FixedKAttention(
        VPolicy("q8",legacy_layout=True),prefix=prefix)).token_ids
    audited=Probe(prefix,targets,sorted({r["page"] for r in diag["rows"]}))
    cache=_copy_prefix(state.cache)
    with audited:
        for step,token in enumerate(base[:max(targets)+1]):
            audited.step=step
            model(torch.tensor([[token]],device=state.last_logits.device),past_key_values=cache,
                  use_cache=True,logits_to_keep=1)
    key=lambda r: (r["decode_step"],r["layer"],r["group"],r["page"])
    idx={key(r):r for r in diag["rows"]}
    joined=[{**idx[key(row)],**row} for row in audited.rows]
    assert len(joined)==len(diag["rows"])
    _write_json(ROOT/"report/diagnostic/0006_dual_bucket.json",
                dict(method="two INT8 centroids, fixed sign of channel 0, 272 B layout, same Quest page quota",
                     snapshot_steps=targets,rows=joined))
    for step in targets:
        rs=[r for r in joined if r["decode_step"]==step]
        focus=[r for r in rs if max(r["mass"])>=.05 and r["centroid8_rank"]>r["centroid8_budget"]]
        print(step,len(focus),sum(r["dual_rank"]<=r["dual_budget"] for r in focus))
        for r in focus:
            if max(r["mass"])>.8:
                print(r["layer"],r["group"],r["page"],max(r["mass"]),r["dual_rank"],r["dual_budget"])


if __name__=="__main__":
    main()
