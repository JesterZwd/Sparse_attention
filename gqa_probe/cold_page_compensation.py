"""Small charged probe for the estimated cold-page mass after the oracle screen."""
from dataclasses import dataclass
from functools import lru_cache
from math import log

import torch

from .fixed_k_v import burst_round, gather_tokens, ranked_pages, vector_burst
from .page_read_budget import page_history
from .round6_attention import physical_ledger
from .round7_attention import Attention, Policy
from .round7_single import prepare_single
from .vskip_summary import StaticOnline, pack_codes, quantize_vector, unpack_codes, weights


MEAN_BYTES = 144  # K/V INT4 codes, two FP16 scales, length byte, aligned record.


@lru_cache(maxsize=512)
def reserve_hot(tokens):
    h = page_history(tokens, 32)
    full,tail = vector_burst(32),vector_burst(h.tail_size)
    cap = 2*((h.pages-1)*full + tail)//10
    mean_bytes = burst_round(h.pages*MEAN_BYTES)
    summary = burst_round(h.pages*208)
    for n in range(min(h.pages, (cap-summary-mean_bytes)//(2*full)+1), len(h.protected)-1, -1):
        kv = 2*((n-1)*full+tail)
        total = mean_bytes+summary+burst_round(4*n)+kv
        if total <= cap:
            return h,cap,n,dict(total=total,mean_bursts=mean_bytes//32)
    raise ValueError("cold mean summary leaves less than the protected K/V")


def hybrid_output(hot_logits, hot_values, cold_log_mass, cold_values):
    combined = torch.cat((hot_logits,cold_log_mass),dim=-1).softmax(-1)
    n = hot_logits.shape[-1]
    return torch.bmm(combined[...,:n],hot_values) + torch.bmm(combined[...,n:],cold_values)


def encode(x):
    code,scale = quantize_vector(x,4,0)
    return pack_codes(code,4),scale


def decode(pair):
    code,scale = pair
    return unpack_codes(code,4).float()*scale


@dataclass
class OnlineMeans:
    tokens: int
    sealed: tuple
    partial: tuple

    @classmethod
    def build(cls,key,value):
        t = key.shape[-2]
        full = (t//32)*32
        partial = tuple(x[0,:,full:].float().sum(1,keepdim=True) for x in (key,value))
        sealed = tuple(encode(x[0,:,:full].unflatten(1,(-1,32)).float().mean(2))
                       for x in (key,value))
        return cls(t,sealed,partial)

    def append(self,key,value,tokens):
        if tokens != self.tokens+1:
            raise ValueError("nonconsecutive cold summary update")
        self.tokens = tokens
        self.partial = tuple(z+x[0,:,-1:].float() for z,x in zip(self.partial,(key,value)))
        n = (tokens-1)%32+1
        records = tuple(tuple(torch.cat((x,y),dim=1) for x,y in zip(existing,encode(z/n)))
                        for existing,z in zip(self.sealed,self.partial))
        if n==32:
            self.sealed = records
            self.partial = tuple(torch.zeros_like(x) for x in self.partial)
        return tuple(map(decode,records))


class CompensatedAttention(Attention):
    def __init__(self,prefix,means,gold):
        super().__init__(Policy("cold_mean_estimate"),prefix,gold)
        self.means = means

    def attention(self,module,query,key,value,attention_mask,**kwargs):
        if self.step < 0 or query.shape[-2]!=1:
            return self.original(module,query,key,value,attention_mask,**kwargs)
        g,t,dim = key.shape[1:]
        q = query.reshape(g,-1,dim).float()
        scale = kwargs.get("scaling") or dim**-.5
        layer = module.layer_idx
        h,cap,count,cost = reserve_hot(t)
        if layer not in self.online:
            self.online[layer] = StaticOnline(self.prefix[layer],("q6",))
        summaries = self.online[layer].append(key[0,:,-1],t)["q6"]
        ranks = ranked_pages(weights(q,summaries,"q6",scale),h)
        k,*_ = prepare_single(q,key,ranks,t,scale,"q6",1000,count)
        key_mean,value_mean = self.means[layer].append(key,value,t)
        hot_pages = torch.zeros(g,h.pages,dtype=torch.bool,device=q.device).scatter_(1,k.ids,True)
        cold_log = torch.bmm(q,key_mean.transpose(1,2))*scale
        cold_log[...,:-1] += log(32)
        cold_log[...,-1] += log(h.tail_size)
        cold_log.masked_fill_(hot_pages[:,None],-torch.inf)
        logits = k.logits.flatten(2)
        y = hybrid_output(logits,gather_tokens(value,k.positions),cold_log,value_mean)
        led = physical_ledger(k,h,"q6",torch.ones(g,dtype=torch.bool,device=q.device))
        mean_read = 32*cost["mean_bursts"]
        total = led["total"]+mean_read
        torch._assert_async(((total==cost["total"])&(total<=cap)).all(),
                            "cold mean physical read cap exceeded")
        for name,counted in led.items():
            self._add("read" if name=="total" else name,total if name=="total" else counted)
        self._add("cold_mean_read_bytes",mean_read*g)
        self._add("cold_mean_write_bytes",MEAN_BYTES*g)
        self._add("cold_mean_prefill_write_bytes",MEAN_BYTES*g*((t-1)//32) if self.step==0 else 0)
        self._add("cold_mean_channel_adds",2*g*dim)
        self._add("over_budget_group_steps",total>cap)
        self._add("decode_layer_calls",1)
        self._add("kv_group_calls",g)
        self._add("scorer_mac",g*q.shape[1]*h.pages*dim*2)
        self._add("qk_mac",led["k_tokens"]*q.shape[1]*dim)
        self._add("pv_mac",(led["v_tokens"]+(h.pages-count))*q.shape[1]*dim)
        self._record_gold_aggregate(hot_pages,g,t,q.device)
        return self.output(query,y)
