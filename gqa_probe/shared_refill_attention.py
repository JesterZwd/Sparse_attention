"""Closed-loop mask reference with shared-refill counters."""
import torch

from .fixed_k_v import gather_tokens, masked_output, probabilities, ranked_pages
from .oracle_attention import OracleAttention
from .shared_refill import PAGE, ledger, quota, run_shared
from .shared_refill_audit import snapshot_rows
from .vskip_summary import StaticOnline, weights


SAMPLE_LAYERS=(0,7,14,21,28,35)
SAMPLE_STEPS=(0,1,3,7,15,31,63)


class SharedAttention(OracleAttention):
    def __init__(self,policy,prefix,*,gold_spans=(),sample=False,shadow=(),gold_by_target=()):
        super().__init__("dense",gold_spans=gold_spans,page_size=PAGE)
        self.policy=policy
        self.prefix=prefix
        self.sample=sample
        self.shadow=tuple(shadow)
        self.gold_by_target=tuple(set(pages) for pages in gold_by_target)
        self.online={}
        self.rows=[]
        self.peak=None
        self.peak_total_active=None

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step<0 or query.shape[-2]!=1:
            return self.original(module,query,key,value,attention_mask,**kwargs)
        layer=module.layer_idx
        n=key.shape[-2]
        h,cap,initial=quota(n,self.policy.budget_bp)
        g,d=key.shape[1],key.shape[-1]
        q=query.reshape(g,-1,d).float()
        scaling=kwargs.get("scaling") or d**-.5
        if layer not in self.online:
            self.online[layer]=StaticOnline(self.prefix[layer],("q6",))
        record=self.online[layer].append(key[0,:,-1],n)["q6"]
        rank=ranked_pages(weights(q,record,"q6",scaling),h)
        observe=self.sample and self.step in SAMPLE_STEPS and layer in SAMPLE_LAYERS
        out=run_shared(self.policy,q,key,rank,n,scaling,audit=observe)
        k=out.final
        values=gather_tokens(value,k.positions)
        actual=masked_output(k.logits,values,k.mask)
        led=ledger(k,h)
        torch._assert_async((led["total"]<=cap).all(),"read ceiling not respected")
        pv=k.mask.sum((1,2,3))*d
        head_count=q.shape[1]
        for label,amount in led.items():
            self._add(label,amount*g if isinstance(amount,int) else amount)
        self._add("initial_k_pages",initial*g)
        self._add("added_k_pages",out.additional)
        self._add("qk_mac",led["k_tokens"]*head_count*d)
        self._add("pv_mac",pv)
        self._add("kv_group_calls",g)
        self._add("recomputed",out.recomputed)
        self._add("fallback",out.fallback)
        self._add("groups_with_added",out.additional>0)
        maximum=led["k_pages"].max()
        total=led["k_pages"].sum()
        self.peak=maximum if self.peak is None else torch.maximum(maximum,self.peak)
        self.peak_total_active=total if self.peak_total_active is None else torch.maximum(total,self.peak_total_active)
        selected_mask=torch.zeros((g,h.pages),device=q.device,dtype=torch.bool).scatter_(1,k.ids,k.mask.any((1,3)))
        self._record_gold_aggregate(selected_mask,g,n,q.device)
        if observe:
            self._observe(self.policy,out,values,actual,rank,led,h,layer,pv,"closed_loop")
            for alternative in self.shadow:
                alternative_out=run_shared(alternative,q,key,rank,n,scaling,audit=True)
                p=alternative_out.final
                alt_v=gather_tokens(value,p.positions)
                alt_pv=p.mask.sum((1,2,3))*d
                alt_attn=masked_output(p.logits,alt_v,p.mask)
                self._observe(alternative,alternative_out,alt_v,alt_attn,rank,ledger(p,h),
                              h,layer,alt_pv,"static_10_common_qkv")
        self._add("decode_layer_calls",1)
        return actual.reshape(1,query.shape[1],1,d).to(query.dtype).transpose(1,2).contiguous(),None

    def _observe(self,policy,out,values,actual,rank,led,h,layer,pv,basis):
        first,final=out.first,out.final
        first_values=values[:,:out.initial_count*PAGE]
        unpruned=masked_output(first.logits,first_values,first.valid[:,None].expand_as(first.logits))
        first_out=masked_output(first.logits,first_values,first.mask)
        final_full=masked_output(final.logits,values,final.valid[:,None].expand_as(final.logits))
        first_loss=(probabilities(first.logits)*~first.mask).sum((2,3))
        final_loss=(probabilities(final.logits)*~final.mask).sum((2,3))
        rows=snapshot_rows(out,h,rank,led,(first_out-unpruned).norm(dim=-1),
                           (actual-final_full).norm(dim=-1),first_loss,final_loss,layer,self.step,policy,pv)
        ids=final.ids.cpu().tolist()
        live=final.valid.any(-1).cpu().tolist()
        kept=final.mask.any((1,3)).cpu().tolist()
        for g,r in enumerate(rows):
            selected={p for p,f in zip(ids[g],live[g]) if f}
            read={p for p,f in zip(ids[g],kept[g]) if f}
            r["gold_targets_with_selected_k_page_omitted_in_v"]=[
                i for i,pages in enumerate(self.gold_by_target) if (pages&selected)-read]
        self.rows.extend(dict(policy=policy.name,basis=basis,**r) for r in rows)

    def stats(self):
        result=super().stats()
        result["peak_k_pages_per_group"]=self.peak.item() if self.peak is not None else 0
        result["peak_active_pages_across_groups"]=self.peak_total_active.item() if self.peak_total_active is not None else 0
        return result
