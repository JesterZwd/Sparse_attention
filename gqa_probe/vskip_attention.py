"""Closed-loop calibration plus same-K shadow evaluation."""
from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np
import torch

from .fixed_k_v import (burst_round, byte_quota, candidate_qk, gather_tokens, keep_mask,
                       masked_output, micro_order, page_keep, probabilities, ranked_pages,
                       traffic, vector_burst)
from .oracle_attention import OracleAttention, grouped_logits, grouped_output, torch_page_mass
from .page_read_budget import page_history
from .vskip_replay import progressive, replay
from .vskip_summary import METHOD_BYTES, StaticOnline, weights


EPS = (.005, .01, .02, .05)
LAMBDAS = (.0001, .001, .01, .03, .1, .25, .5)
LAYERS = (0, 7, 14, 21, 28, 35)
STEPS = (0, 1, 3, 7, 15, 31, 63)


@dataclass(frozen=True)
class VPolicy:
    scorer: str
    page: int = 32
    percent: int = 10
    mode: str = "all"
    threshold: float = 0
    equal_pages: bool = False
    legacy_layout: bool = False
    refill: bool = False

    @property
    def name(self):
        return f"{self.scorer}_P{self.page}_b{self.percent}_{self.mode}_{self.threshold:g}" + (
            "_same_pages" if self.equal_pages else "") + ("_refill" if self.refill else "")

    def count(self, tokens):
        scorer = "q8" if self.equal_pages or self.scorer not in METHOD_BYTES else self.scorer
        return byte_quota(tokens, self.page, scorer, self.percent, burst=not self.legacy_layout)


class FixedKAttention(OracleAttention):
    def __init__(self, policy, *, gold_spans=(), prefix=None, audit=False, residual=False):
        super().__init__("dense", gold_spans=gold_spans, page_size=policy.page)
        self.policy = policy
        self.prefix = prefix
        self.online = {}
        self.audit = audit
        self.residual = residual
        self.traffic = Counter()
        self.curve = []
        self.schedule = []
        self.progressive = []
        self.zero_max_error = 0.
        self.exact_shadow = []

    def _replay(self, ids, slots, mask, h, layer, method, rule, batch):
        p, kept = ids.gather(-1, slots).cpu().tolist(), mask.gather(1, slots).cpu().tolist()
        for group, (pages, active) in enumerate(zip(p, kept)):
            row = replay(pages, active, h.tokens, h.page_size, method, batch=batch)
            width = batch or len(pages)
            self.schedule.append(dict(layer=layer, step=self.step, group=group, method=rule,
                                      qk_pages=len(pages), kept_pages=sum(active),
                                      page_mass_table_bytes=width * 4 * 4,
                                      logits_fp16_bytes=width * h.page_size * 4 * 2,
                                      accumulator_fp32_bytes=4 * (2+128) * 4,
                                      protection_mask_bytes=math.ceil(width/8), **row))

    def audit_state(self, logits, p, values, ids, traversal, valid, protected, h, layer, method, records, query, scaling):
        full = keep_mask(logits, p, valid, protected, "all", 0)
        result = masked_output(logits, values, full)
        test_zero = masked_output(logits, values, keep_mask(logits, p, valid, protected, "mass", 0))
        zero = (result-test_zero).abs().amax()
        self.exact_shadow.append(zero)
        variants = [("all", 0, full, None, 0)]
        variants.extend((mode, eps, keep_mask(logits, p, valid, protected, mode, eps, traversal), None, 0)
                        for mode, thresholds in (("mass", EPS), ("sparta", EPS), ("blasst", LAMBDAS))
                        for eps in thresholds)
        mass = p.sum(-1)
        for batch in (1,4,8):
            for by_address in (False, True):
                slots = micro_order(traversal, ids, batch, by_address)
                kept = page_keep(mass, protected, .02, order=slots, batch=batch)
                mask = (kept[..., None] & valid)[:, None].expand_as(logits)
                kind = f"micro_{batch}_{'addr' if by_address else 'rank'}"
                variants.append((kind, .02, mask, slots, batch))
        stats = []
        names = []
        norm = result.norm(dim=-1).clamp_min(1e-12)
        for kind, eps, mask, slots, batch in variants:
            lost = (p * ~mask).sum((2,3))
            if kind in ("mass", "sparta") or kind.startswith("micro"):
                torch._assert_async((lost <= eps+3e-6).all(), "final head mass exceeds cap")
            error = (masked_output(logits, values, mask) - result).norm(dim=-1)
            v, k, total, keep, pv = traffic(mask, valid, h, method)
            stats.append(torch.cat((torch.stack((v,k,total,keep,pv,lost.amax(1),
                                                 protected.sum(1),valid.any(-1).sum(1)),dim=1),
                                    error, error/norm), dim=1))
            names.append((kind,eps))
            if kind == "all" or (kind == "mass" and eps == .02) or kind.startswith("micro"):
                replay_slots = traversal if slots is None else slots
                self._replay(ids, replay_slots, mask.any(1).any(-1), h, layer, method, kind, batch)
        packed = torch.stack(stats).cpu().tolist()
        for (kind,eps), group_rows in zip(names,packed):
            for g, r in enumerate(group_rows):
                v,k,total,kp,pv,loss,prot,avail = r[:8]
                heads = p.shape[1]
                self.curve.append(dict(layer=layer,step=self.step,group=g,mode=kind,threshold=eps,
                                       v_bytes=int(v),k_bytes=int(k),total_bytes=int(total),
                                       v_pages=int(kp),k_pages=int(avail),protected_pages=int(prot),
                                       lost_max=loss,pv_mac=int(pv),absolute=r[8:8+heads],relative=r[8+heads:]))
        if self.residual and layer in (0,14,35) and self.step in (0,3):
            a,sa,b,sb = (x.cpu().numpy() for x in records)
            for g,q in enumerate(query.cpu().numpy()):
                reads,fallback,exact = progressive(q,a[g].view(np.int8),sa[g],b[g].view(np.int8),
                                                    sb[g],h.protected,ids.shape[1],scaling)
                self.progressive.append(dict(layer=layer,step=self.step,group=g,pages=h.pages,
                                             residual_pages=reads,fallback=fallback,exact=exact,
                                             coarse_burst_bytes=burst_round(h.pages*208),
                                             residual_burst_bytes=reads*64,
                                             int8_burst_bytes=burst_round(h.pages*272)))

    def attention(self, module, query, key, value, attention_mask, **kwargs):
        if self.step < 0 or query.shape[-2] != 1:
            return self.original(module, query, key, value, attention_mask, **kwargs)
        layer = module.layer_idx
        policy = self.policy
        h = page_history(key.shape[-2], policy.page)
        groups, dim = key.shape[1], key.shape[-1]
        q = query.reshape(groups,-1,dim).float()
        scaling = kwargs.get("scaling") or dim**-.5
        scorer = policy.scorer
        mask = None
        if scorer in ("dense", "oracle"):
            logits = grouped_logits(query, key, scaling)
            if scorer == "oracle":
                chosen = ranked_pages(torch_page_mass(logits, h.page_size).sum(-1), h)[:, :policy.count(h.tokens)]
                mask = torch.zeros_like(logits[:,0,::h.page_size],dtype=torch.bool).scatter_(1,chosen,True)
            out = grouped_output(logits, value, mask, h.page_size)
        else:
            if layer not in self.online:
                self.online[layer] = StaticOnline(self.prefix[layer],(scorer,))
            metadata = self.online[layer].append(key[0,:,-1],h.tokens)[scorer]
            rankings = ranked_pages(weights(q, metadata, scorer, scaling), h)
            count = policy.count(h.tokens)
            ids, permutation = rankings[:,:count].sort(dim=-1)
            traversal = permutation.argsort(dim=-1)
            logits, pos, valid, forced = candidate_qk(q,key,ids,h,scaling)
            p = probabilities(logits) if policy.mode != "all" or self.audit else None
            pv = keep_mask(logits,p,valid,forced,policy.mode,policy.threshold,traversal)
            values = gather_tokens(value,pos)
            if self.audit and layer in LAYERS and self.step in STEPS:
                self.audit_state(logits,p,values,ids,traversal,valid,forced,h,layer,scorer,metadata,q,scaling)
            if policy.refill:
                (ids, logits, pos, valid, pv) = self.refill(h,q,key,scaling,rankings,ids,logits,pos,valid,pv)
                values = gather_tokens(value,pos)
            out = masked_output(logits,values,pv)
            v,k,total,v_pages,pv_mac = traffic(pv,valid,h,scorer)
            if not policy.equal_pages and not policy.legacy_layout:
                dense_burst = 2*((h.pages-1)*vector_burst(h.page_size)+vector_burst(h.tail_size))
                torch._assert_async((total <= dense_burst*policy.percent//100).all(), "burst ceiling exceeded")
            for label,x in zip(("v_bytes","k_bytes","read_bytes","v_pages","pv_mac"),(v,k,total,v_pages,pv_mac)):
                self._add(label,x)
            self._add("k_pages",valid.any(-1).sum())
            self.traffic["summary_bytes"] += groups*burst_round(h.pages*METHOD_BYTES[scorer])
            self.traffic["summary_write_bytes"] += groups*METHOD_BYTES[scorer]
            mask = torch.zeros((groups,h.pages),dtype=torch.bool,device=q.device)
            mask.scatter_(1,ids,pv.any(1).any(-1))
        self._record_gold_aggregate(mask,groups,h.tokens,q.device)
        self._add("decode_layer_calls",1)
        return out.reshape(1,query.shape[1],1,dim).to(query.dtype).transpose(1,2).contiguous(), None

    def refill(self,h,q,key,scaling,rankings,ids,logits,pos,valid,keep):
        policy = self.policy
        count = ids.shape[-1]
        dense = 2*((h.pages-1)*vector_burst(h.page_size)+vector_burst(h.tail_size))
        cap = dense*policy.percent//100
        _,_,used,_,_ = traffic(keep,valid,h,policy.scorer)
        full = vector_burst(h.page_size)
        # Reserve a full K+V page and the largest possible index burst increment.
        room = ((cap-used)//(2*full+32)).clamp_min(0)
        upper = min(h.pages-count, (count-len(h.protected))//2+2)
        if upper <= 0:
            return ids,logits,pos,valid,keep
        extra_ids = rankings[:,count:count+upper]
        active = torch.arange(upper,device=q.device)[None,:] < room[:,None]
        extra_logits, extra_pos, extra_valid, _ = candidate_qk(q,key,extra_ids,h,scaling,active)
        new = torch.cat((logits,extra_logits),dim=2)
        p = probabilities(new)
        per_page = p.sum(-1)
        anchored = (p[...,:count,:] * ~keep).sum((2,3))
        # One reservation pass. Once all extra K is visible, release V for each
        # added page that fits the remaining per-head cap.
        masks = []
        for j in range(upper):
            w = per_page[...,count+j]
            drop = (anchored+w <= policy.threshold-1e-7).all(1) & active[:,j]
            anchored = anchored+w*drop[:,None]
            masks.append(~drop)
        extra_keep = (torch.stack(masks,1)[...,None] & extra_valid)[:,None].expand_as(extra_logits)
        return (torch.cat((ids,extra_ids),1),new,torch.cat((pos,extra_pos),1),
                torch.cat((valid,extra_valid),1),torch.cat((keep,extra_keep),2))

    def stats(self):
        stats = super().stats()
        for k in ("v_bytes","k_bytes","read_bytes","v_pages","k_pages","pv_mac"):
            stats.setdefault(k,0)
        stats.update(self.traffic)
        stats["zero_max_error"] = max((x.item() for x in self.exact_shadow),default=None)
        return stats
