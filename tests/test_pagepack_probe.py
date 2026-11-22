import torch

from gqa_probe.pagepack_probe import vector_bytes, segment_logit_span


def test_burst_planes_include_short_tail():
    x=torch.tensor([1,4,8,32])
    assert vector_bytes(x).tolist()==[128,320,608,2432]


def test_segment_span_excludes_invalid_tail_and_reports_evidence():
    z=torch.tensor([[[[3.,-1.,4.,float("-inf")]]]])
    m=torch.tensor([[[0,0,1,0]]])
    n=torch.tensor([[[2,1,0,0]]])
    valid=torch.tensor([[[True,True,True,False]]])
    span, merged=segment_logit_span(z,m,n,valid)
    assert span[0,0,0,0]==4.
    assert merged.sum()==1
