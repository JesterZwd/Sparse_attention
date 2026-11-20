"""Construct coalesced bursts and replay explicit read dependencies."""
import ctypes
from functools import lru_cache
import os
from pathlib import Path
import subprocess

import numpy as np
from .fixed_k_v import burst_round,vector_burst
from .shared_refill_audit import runs


def compiled():
    target=Path(".cache/round6/lpddr5.so")
    target.parent.mkdir(parents=True,exist_ok=True)
    src=Path(__file__).resolve().parents[1]/"sim/round6_lpddr5.cpp"
    if not target.is_file() or target.stat().st_mtime_ns<src.stat().st_mtime_ns:
        tmp=target.with_name(f"lpddr5-{os.getpid()}.so")
        subprocess.run(["g++","-std=c++17","-O3","-shared","-fPIC",str(src),"-o",str(tmp)],check=True)
        tmp.replace(target)
    return target.resolve()


@lru_cache(maxsize=1)
def engine():
    dll=ctypes.CDLL(str(compiled()))
    fn=dll.simulate
    arg=np.ctypeslib.ndpointer(dtype=np.int64,ndim=1,flags="C_CONTIGUOUS")
    fn.argtypes=[arg,arg,arg,arg,ctypes.c_int64,arg]
    fn.restype=None
    return fn


def merge(seq):
    result=[]
    for addr,size in seq:
        if not size:
            continue
        if addr%32 or size%32:
            raise ValueError("address and request must align to a 32 B burst")
        if result and result[-1][0]+result[-1][1]==addr:
            a,s=result[-1]
            result[-1]=(a,s+size)
        else:
            result.append((addr,size))
    return result


def read_phases(phases, wait=(0,0,0,0), descriptor=2):
    flat=[run for phase in phases for run in phase]
    starts=np.array([a//32 for a,_ in flat],dtype=np.int64)
    sizes=np.array([s//32 for _,s in flat],dtype=np.int64)
    bounds=np.array([0,*np.cumsum([len(p) for p in phases])],dtype=np.int64)
    before=np.array(wait,dtype=np.int64)
    out=np.zeros(8,dtype=np.int64)
    engine()(starts,sizes,bounds,before,descriptor,out)
    assert int(out[4])==int(sizes.sum())
    return [int(x) for x in out]


def expansion(mask,width,valid):
    if width==1:
        return mask&valid
    return np.repeat(mask.reshape(-1,width).any(-1),width)&valid


def value_runs(mask,count,base):
    regions=((base,np.repeat(mask,2)),
             (base+64*count,mask.reshape(-1,4).any(-1)),
             (base+64*count+burst_round(8*count),mask.reshape(-1,8).any(-1)))
    return [(a+32*l,32*(r-l)) for a,part in regions for l,r in runs(part)]


def transaction_trace(h,initial,added,mask,meta_bytes,*,width=1,index=True):
    """No repeated K page or V bursts; reads are in phase order."""
    ids=[int(p) for p in (*initial,*added)]
    if len(ids)!=len(set(ids)) or not set(mask).issubset(ids):
        raise ValueError("duplicate K or V outside the candidate set")
    stride=vector_burst(h.page_size)
    def n(p):
        return min(h.page_size,h.tokens-p*h.page_size)
    def k(seq):
        return merge(((1<<22)+int(p)*stride,vector_burst(n(int(p)))) for p in seq)
    p0=merge([(0,burst_round(meta_bytes)),(1<<20,burst_round(4*len(ids)) if index else 0)])
    rr=[]
    expanded=0
    for p,pv in sorted(mask.items()):
        old=np.asarray(pv,dtype=bool)
        val=np.arange(h.page_size)<n(p)
        new=expansion(old,width,val)
        expanded+=int(new.sum()-old.sum())
        rr.extend(value_runs(new,n(p),(1<<25)+p*stride))
    return [p0,k(initial),k(added),merge(rr)],expanded


def simulate(h,initial,added,mask,meta_bytes,wait,*,width=1,index=True):
    phases,expanded=transaction_trace(h,initial,added,mask,meta_bytes,width=width,index=index)
    timing=read_phases(phases,wait)
    raw=read_phases(phases,(0,0,0,0))
    return dict(bursts=timing[4],reads_bytes=32*timing[4],lpddr_ck=timing[3],
                lpddr_reads_and_drains_ck=raw[3],summary_index_ck=timing[0],
                before_first_v_ck=timing[2]+wait[3],added_k_bursts=sum(s//32 for _,s in phases[2]),
                read_runs=sum(map(len,phases)),v_runs=len(phases[3]),
                expanded_tokens=expanded,row_hits=timing[5],activations=timing[6],
                extra_v_bytes=sum(s for _,s in phases[3])-sum(s for _,s in
                     transaction_trace(h,initial,added,mask,meta_bytes,width=1,index=index)[0][3]) if width>1 else 0)
