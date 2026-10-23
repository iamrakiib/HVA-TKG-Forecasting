#!/usr/bin/env python3
import argparse, copy, json, random
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from revision_common import *

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)
def precompute(scores,q,h,k,thr):
    ids=topk_indices(scores,k); X=np.zeros((len(q),ids.shape[1],4),np.float32); sr=h['sr']
    for i,(s,r,_o,t) in enumerate(q):
        m=sr.get((int(s),int(r)),{})
        for j,c in enumerate(ids[i]): X[i,j]=exact_features(m.get(int(c),[]),int(t),thr)
    return ids,X

def apply_numpy(scores,ids,X,w):
    out=scores.copy(); delta=X@w.astype(np.float32); out[np.arange(len(out))[:,None],ids]+=delta; return out
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--valid-dump',required=True); ap.add_argument('--test-dump',required=True); ap.add_argument('--topk',type=int,default=100); ap.add_argument('--stale-threshold',type=int,default=10); ap.add_argument('--dev-frac',type=float,default=.2); ap.add_argument('--epochs',type=int,default=40); ap.add_argument('--patience',type=int,default=6); ap.add_argument('--lr',type=float,default=.03); ap.add_argument('--weight-decay',type=float,default=1e-4); ap.add_argument('--pairwise-weight',type=float,default=.25); ap.add_argument('--margin',type=float,default=.2); ap.add_argument('--delta-reg',type=float,default=1e-4); ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--out-dir',default='results/linear_same_features'); a=ap.parse_args(); seed_all(a.seed); device=torch.device('cpu')
_,nr,sp,vh,vf=context(a.data_root,a.dataset,'valid'); vs,vtr,ve=load_score_dump(a.valid_dump); vq=canonicalize_queries(vtr,ve,nr); _,_,_,th,tf=context(a.data_root,a.dataset,'test'); ts,ttr,te=load_score_dump(a.test_dump); tq=canonicalize_queries(ttr,te,nr)
vid,vX=precompute(vs,vq,vh,a.topk,a.stale_threshold); tid,tX=precompute(ts,tq,th,a.topk,a.stale_threshold)
order=np.argsort(vq[:,3],kind='stable'); cut=max(1,min(len(order),int(round(len(order)*(1-a.dev_frac))))); tridx=order[:cut]; didx=order[cut:]
w=torch.nn.Parameter(torch.zeros(4,dtype=torch.float32)); opt=torch.optim.AdamW([w],lr=a.lr,weight_decay=a.weight_decay); best=None; bestm=-1; bad=0; log=[]
for ep in range(1,a.epochs+1):
    perm=np.random.permutation(tridx); total=0.0
    for st in range(0,len(perm),a.batch_size):
        ix=perm[st:st+a.batch_size]; score=torch.tensor(vs[ix],dtype=torch.float32); ids=torch.tensor(vid[ix],dtype=torch.long); X=torch.tensor(vX[ix],dtype=torch.float32); gold=torch.tensor(vq[ix,2],dtype=torch.long); delta=X@w; adj=score.clone(); adj.scatter_add_(1,ids,delta); ce=F.cross_entropy(adj,gold); gs=adj.gather(1,gold[:,None]).squeeze(1); neg=adj.clone(); neg.scatter_(1,gold[:,None],float('-inf')); hard=neg.max(1).values; pair=F.relu(a.margin-(gs-hard)).mean(); reg=delta.pow(2).mean(); loss=ce+a.pairwise_weight*pair+a.delta_reg*reg; opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss.detach().item())*len(ix)
    if len(didx):
        da=apply_numpy(vs[didx],vid[didx],vX[didx],w.detach().numpy()); dm=evaluate_scores(da,vq[didx],vf)['MRR']
    else: dm=-total/max(len(tridx),1)
    log.append({'epoch':ep,'train_loss':total/max(len(tridx),1),'dev_MRR':dm})
    if dm>bestm: bestm=dm; best=w.detach().clone(); bad=0
    else: bad+=1
    if ep>=5 and bad>=a.patience: break
if best is not None: w.data.copy_(best)
wn=w.detach().numpy(); vadj=apply_numpy(vs,vid,vX,wn); tadj=apply_numpy(ts,tid,tX,wn); outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
res={'dataset':a.dataset,'method':'Linear_same_HVA_exact_features','seed':a.seed,'topk':a.topk,'stale_threshold':a.stale_threshold,'weights':{'occurrence':float(wn[0]),'recency':float(wn[1]),'log_frequency':float(wn[2]),'stale':float(wn[3])},'valid_overall':evaluate_scores(vadj,vq,vf),'test_overall':evaluate_scores(tadj,tq,tf),'train_log':log,'note':'Backbone frozen. Only four residual weights are learned from VALID; TEST is untouched until final evaluation.'}; save_json(res,outdir/'linear_results.json'); np.savez_compressed(outdir/'valid_linear_scores.npz',scores=vadj,triples=vq,entity=np.asarray(['object'])); np.savez_compressed(outdir/'test_linear_scores.npz',scores=tadj,triples=tq,entity=np.asarray(['object'])); print(json.dumps(res,indent=2))
