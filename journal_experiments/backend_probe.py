"""Official Mamba2/Mamba3 random-weight smoke-test adapter (not segmentation results)."""
import argparse
import json
import torch
from scan_hsic_pilot import paths, inverse

def run_scans(x, idx, block):
    b,c,h,w=x.shape
    length=h*w
    scan_ids=idx.to(x.device)[None,:,None,:].expand(b,4,c,length)
    input_seq=x.reshape(b,c,length)[:,None].expand(b,4,c,length)
    seq=input_seq.gather(-1,scan_ids).permute(0,1,3,2).reshape(b*4,length,c)
    y=block(seq)
    assert y.shape==seq.shape
    y=y.reshape(b,4,length,c).permute(0,1,3,2)
    inv_ids=inverse(idx).to(x.device)[None,:,None,:].expand_as(scan_ids)
    return y.gather(-1,inv_ids).mean(1).reshape(b,c,h,w)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--backend',choices=('mamba2','mamba3'),default='mamba2')
    ap.add_argument('--dim',type=int,default=64)
    ap.add_argument('--size',type=int,default=8)
    args=ap.parse_args()
    if not torch.cuda.is_available():
        print(json.dumps({'status':'SKIPPED','backend':args.backend,'reason':'No CUDA GPU available'}))
        return
    from mamba_ssm import Mamba2,Mamba3
    cls=Mamba2 if args.backend=='mamba2' else Mamba3
    kw={'d_model':args.dim,'d_state':64,'expand':2,'headdim':64}
    if args.backend=='mamba3':
        kw.update({'d_state':128,'chunk_size':64,'is_mimo':False})
    m=cls(**kw).cuda()
    if args.backend=='mamba3':
        m=m.to(dtype=torch.bfloat16)
    x=torch.randn(1,args.dim,args.size,args.size,device='cuda',dtype=next(m.parameters()).dtype,requires_grad=True)
    result={}
    for mode in ('cross','topoa'):
        m.zero_grad(set_to_none=True)
        if x.grad is not None: x.grad.zero_()
        y=run_scans(x,paths(args.size,args.size,mode),m)
        y.float().square().mean().backward()
        result[mode]={'shape':list(y.shape),'finite_output':bool(torch.isfinite(y).all().item()),'finite_gradient':bool(torch.isfinite(x.grad).all().item())}
    print(json.dumps({'status':'PASS','backend':args.backend,'checks':result},indent=2))

if __name__=='__main__':main()
