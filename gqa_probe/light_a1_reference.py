"""Integer reference used when evaluating an A1 normalization substitution."""
from math import log2
import numpy as np

TABLE=np.rint(2**20 * 2.**(-np.arange(16)/16)).astype(np.int64)
RECIP=np.rint(2**16/(1+(np.arange(256)+.5)/256)).astype(np.int64)


def rne_integer(n,d):
    q,r=np.divmod(n,d)
    return q+((2*r>d)|((2*r==d)&((q&1)!=0)))


def light_probabilities(scores,valid=None,*,exact=False):
    """[H,J] scores; all valid pages participate including forced protection."""
    vals=np.asarray(scores,dtype=np.float64)
    if vals.ndim!=2:
        raise ValueError("head by page scores expected")
    visible=np.broadcast_to(np.asarray(valid,dtype=bool),vals.shape) if valid is not None else np.ones(vals.shape,dtype=bool)
    if not visible.any(-1).all() or not np.isfinite(vals[visible]).all():
        raise ValueError("every head needs finite valid page scores")
    fixed=np.rint(np.where(visible,vals,0)*(16*log2(np.e))).astype(np.int64)
    maxima=np.where(visible,fixed,np.iinfo(np.int64).min).max(-1,keepdims=True)
    diff=np.where(visible,maxima-fixed,321)
    use=visible&(diff<=320)
    x=np.clip(diff,0,320)
    code=np.where(use,TABLE[x&15]>>(x>>4),0)
    z=code.sum(-1,keepdims=True,dtype=np.int64)
    if exact:
        return code,rne_integer(code*(1<<20),z)
    a=np.array([int(v).bit_length()-1 for v in z[:,0]],dtype=np.int64)[:,None]
    j=(z>>(a-8))&255
    return code,rne_integer(code*RECIP[j],1<<(a-4))
