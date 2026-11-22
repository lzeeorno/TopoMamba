"""CPU scan/HSIC falsification probe. No trained segmentation evaluation."""
import json
import math
import torch
import torch.nn.functional as F

def paths(h, w, mode):
    grid = torch.arange(h*w).reshape(h,w)
    if mode == 'cross':
        a,b = grid.flatten(), grid.T.contiguous().flatten()
    elif mode == 'topoa':
        da, an = [], []
        for s in range(h+w-1):
            row, anti = [], []
            for i in range(h):
                j=s-i
                if 0<=j<w:
                    row.append(i*w+j)
                    anti.append(i*w+w-1-j)
            if s%2:
                row.reverse()
                anti.reverse()
            da.extend(row)
            an.extend(anti)
        a,b = torch.tensor(da), torch.tensor(an)
    else:
        raise ValueError(mode)
    return torch.stack((a,b,a.flip(0),b.flip(0)))

def inverse(paths):
    inv = torch.empty_like(paths)
    inv.scatter_(1, paths, torch.arange(paths.shape[1]).expand_as(paths))
    return inv

def gap(mask, order):
    h,w=mask.shape
    inv = inverse(order)
    gs = []
    for y,x in torch.nonzero(mask).tolist():
        for dy,dx in ((0,1),(1,-1),(1,0),(1,1)):
            yy,xx = y+dy,x+dx
            if 0<=yy<h and 0<=xx<w and mask[yy,xx]:
                gs.append((inv[:,y*w+x]-inv[:,yy*w+xx]).abs().min().item())
    return round(sum(gs)/len(gs),3) if gs else None

def centered_rbf(x):
    d=torch.cdist(x.float(),x.float()).square()
    bandwidth=d.flatten(1).median(-1).values.clamp_min(1e-6)
    k=torch.exp(-d/(2*bandwidth[:,None,None]))
    return k-k.mean(-1,keepdim=True)-k.mean(-2,keepdim=True)+k.mean((-1,-2),keepdim=True)

def hsic(a,b,seed=7):
    B,C,L=a.shape
    g=torch.Generator().manual_seed(seed)
    p=F.normalize(torch.randn(L,min(16,L),generator=g),dim=0)
    x=F.normalize(a@p,dim=-1)
    y=F.normalize(b@p,dim=-1)
    k,l=centered_rbf(x),centered_rbf(y)
    return (k*l).sum((-1,-2))/max((C-1)**2,1)

def main():
    torch.manual_seed(2026)
    h=w=32
    masks={}
    z=torch.zeros(h,w,dtype=torch.bool);z[h//2,3:-3]=True;masks['horizontal']=z
    z=torch.zeros(h,w,dtype=torch.bool);z[3:-3,w//2]=True;masks['vertical']=z
    z=torch.zeros(h,w,dtype=torch.bool)
    for i in range(3,29):z[i,i]=True
    masks['45deg']=z
    z=torch.zeros(h,w,dtype=torch.bool)
    for x in range(3,29):
        y=round(h/2+5*math.sin((x-3)/26*2*math.pi))
        z[y,x]=True
    masks['curved']=z
    geo={name:{mode:gap(m,paths(h,w,mode)) for mode in ('cross','topoa')} for name,m in masks.items()}
    for mode in ('cross','topoa'):
        p=paths(9,11,mode)
        assert torch.equal(torch.sort(p,dim=1).values,torch.arange(99).expand(4,99))
        assert torch.equal(inverse(p).gather(1,p),torch.arange(99).expand(4,99))
    a=torch.randn(8,32,256); corr=a+.02*torch.randn_like(a); indep=torch.randn_like(a)
    vals=torch.cat((hsic(a,corr),hsic(a,indep)))
    wgt=torch.sigmoid(.5*vals/1.5)
    report={'status':'CPU synthetic only','geometry_mean_min_sequence_gap':geo,
            'hsic_correlated_mean':hsic(a,corr).mean().item(),
            'hsic_independent_mean':hsic(a,indep).mean().item(),
            'gate_std':wgt.std().item(),
            'paper_TopoA_mean':(.2+.8*wgt).mean().item(),
            'public_2D_code_TopoA_mean':(.2+.8*(1-wgt)).mean().item()}
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
