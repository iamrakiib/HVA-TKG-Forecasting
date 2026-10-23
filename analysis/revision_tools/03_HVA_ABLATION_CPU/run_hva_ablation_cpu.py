#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path
import numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from hva_core.history_utils import *
from hva_core.metrics import evaluate_filtered, evaluate_diagnostics

def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)

def feature4(times,t,thr):
    lt=last_time_before(times,int(t))
    if lt is None: return (0.,0.,0.,0.)
    gap=max(0,int(t)-int(lt)); freq=freq_before(times,int(t)); return (1.,1./(1.+float(gap)),float(np.log1p(freq)),1. if gap>thr else 0.)

def build_features(q,cids,h,thr,mask):
    q=np.asarray(q); c=np.asarray(cids); F=np.zeros((len(q),c.shape[1],4),np.float32); sr=h['sr']
    for i,(s,r,_o,t) in enumerate(q):
        m=sr.get((int(s),int(r)),{})
        for j,x in enumerate(c[i]): F[i,j]=feature4(m.get(int(x),[]),int(t),thr)
    F*=np.asarray(mask,dtype=np.float32)[None,None,:]; return torch.tensor(F,dtype=torch.float32)

class Adapter(nn.Module):
    def __init__(self,nrel,hidden=64,redim=16,drop=.1,gamma=.1,disable_rel=False):
        super().__init__(); self.disable_rel=disable_rel; self.rel_emb=nn.Embedding(max(1,int(nrel)),int(redim)); self.mlp=nn.Sequential(nn.Linear(4+redim+3,hidden),nn.ReLU(),nn.Dropout(drop),nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(drop),nn.Linear(hidden,1)); self.gamma_raw=nn.Parameter(torch.tensor(float(gamma)))
    def forward(self,base,rel,feat):
        centered=base-base.mean(1,keepdim=True); gap=base.max(1,keepdim=True).values-base; k=base.shape[1]; rank=torch.arange(k,dtype=torch.float32).view(1,k).expand_as(base)/max(k-1,1); rc=self.rel_emb(rel.clamp_min(0)%self.rel_emb.num_embeddings).unsqueeze(1).expand(-1,k,-1)
        if self.disable_rel: rc=torch.zeros_like(rc)
        x=torch.cat([feat,rc,centered.unsqueeze(-1),gap.unsqueeze(-1),rank.unsqueeze(-1)],-1); d=self.mlp(x).squeeze(-1); g=torch.tanh(self.gamma_raw); return base+g*d,g*d

def topk_torch(x,k): return torch.topk(x.detach(),k=min(k,x.shape[1]),dim=1).indices

def apply(model,scores,q,h,k,thr,mask,batch=128):
    model.eval(); outs=[]
    with torch.no_grad():
        for st in range(0,len(q),batch):
            en=min(len(q),st+batch); x=torch.tensor(scores[st:en],dtype=torch.float32); ids=topk_torch(x,k); base=x.gather(1,ids); qq=np.asarray(q[st:en]).copy(); qq[:,2]=-1; feat=build_features(qq,ids.numpy(),h,thr,mask); rel=torch.tensor(np.asarray(q[st:en])[:,1],dtype=torch.long); adj,_=model(base,rel,feat); full=x.clone(); full.scatter_(1,ids,adj); outs.append(full.numpy())
    return np.concatenate(outs,0)

def train_adapter(model,scores,q,h,filt,args,mask):
    order=np.argsort(q[:,3],kind='stable'); cut=max(1,min(len(order),int(round(len(order)*(1-args.dev_frac))))); ti=order[:cut]; di=order[cut:]; ds=TensorDataset(torch.tensor(scores[ti],dtype=torch.float32),torch.tensor(q[ti],dtype=torch.long)); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay); best=None; bestm=-1; bad=0; log=[]
    for ep in range(1,args.epochs+1):
        model.train(); tot=0.; n=0
        for sc,qt in loader:
            qq=qt.numpy(); gold=qt[:,2].long(); ids=topk_torch(sc,args.topk); base=sc.gather(1,ids); masked=qq.copy(); masked[:,2]=-1; feat=build_features(masked,ids.numpy(),h,args.stale_threshold,mask); rel=qt[:,1].long(); adjtop,delta=model(base,rel,feat); full=sc.clone(); full.scatter_(1,ids,adjtop); ce=F.cross_entropy(full,gold); gs=full.gather(1,gold[:,None]).squeeze(1); neg=full.clone(); neg.scatter_(1,gold[:,None],float('-inf')); hard=neg.max(1).values; pair=F.relu(args.margin-(gs-hard)).mean(); reg=delta.pow(2).mean(); loss=ce+args.pairwise_weight*pair+args.bias_reg*reg; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_norm); opt.step(); tot+=float(loss.detach().item())*len(sc); n+=len(sc)
        dm=None
        if len(di): dm=evaluate_filtered(apply(model,scores[di],q[di],h,args.topk,args.stale_threshold,mask,args.eval_batch_size),q[di],filt)['MRR']
        log.append({'epoch':ep,'train_loss':tot/max(n,1),'dev_MRR':dm})
        score=dm if dm is not None else -tot/max(n,1)
        if score>bestm: bestm=score; best={k:v.detach().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if ep>=args.min_epochs and bad>=args.patience: break
    if best: model.load_state_dict(best)
    return log

def parse_variant(v):
    mask=[1,1,1,1]; disrel=False
    if v=='no_occurrence': mask[0]=0
    elif v=='no_recency': mask[1]=0
    elif v=='no_frequency': mask[2]=0
    elif v=='no_stale': mask[3]=0
    elif v=='no_relation_context': disrel=True
    elif v!='full': raise ValueError(v)
    return mask,disrel

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True,choices=['ICEWS14','ICEWS18','YAGO']); p.add_argument('--data-root',required=True); p.add_argument('--valid-dump',required=True); p.add_argument('--test-dump',required=True); p.add_argument('--variants',default='full,no_occurrence,no_recency,no_frequency,no_stale,no_relation_context'); p.add_argument('--topk',type=int,default=100); p.add_argument('--epochs',type=int,default=12); p.add_argument('--min-epochs',type=int,default=4); p.add_argument('--patience',type=int,default=3); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--eval-batch-size',type=int,default=128); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight-decay',type=float,default=1e-5); p.add_argument('--hidden-dim',type=int,default=64); p.add_argument('--rel-emb-dim',type=int,default=16); p.add_argument('--dropout',type=float,default=.1); p.add_argument('--gamma',type=float,default=.1); p.add_argument('--stale-threshold',type=int,default=10); p.add_argument('--dev-frac',type=float,default=.2); p.add_argument('--pairwise-weight',type=float,default=.25); p.add_argument('--margin',type=float,default=.2); p.add_argument('--bias-reg',type=float,default=1e-4); p.add_argument('--grad-norm',type=float,default=1.0); p.add_argument('--seed',type=int,default=42); p.add_argument('--out-dir',default='results/hva_ablation'); p.add_argument('--save-adjusted',action='store_true'); args=p.parse_args(); seed_all(args.seed)
    data=os.path.join(args.data_root,args.dataset); ne,nr=read_stat(data); sp=read_split_arrays(data); vs,vtr,ve=load_score_dump(args.valid_dump); ts,ttr,te=load_score_dump(args.test_dump); vq=canonicalize_queries(vtr,ve,nr); tq=canonicalize_queries(ttr,te,nr); train_hist=augment_with_inverse([tuple(x) for x in sp['train']],nr); trainvalid=augment_with_inverse([tuple(x) for x in np.concatenate([sp['train'],sp['valid']],0)],nr); vh=build_histories(train_hist); th=build_histories(trainvalid); vf=augment_with_inverse([tuple(x) for x in np.concatenate([sp['train'],sp['valid']],0)],nr); tf=augment_with_inverse([tuple(x) for x in np.concatenate([sp['train'],sp['valid'],sp['test']],0)],nr); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for v in [x.strip() for x in args.variants.split(',') if x.strip()]:
        seed_all(args.seed); mask,disrel=parse_variant(v); model=Adapter(max(nr*2,int(vq[:,1].max())+1,int(tq[:,1].max())+1),args.hidden_dim,args.rel_emb_dim,args.dropout,args.gamma,disrel); log=train_adapter(model,vs,vq,vh,vf,args,mask); va=apply(model,vs,vq,vh,args.topk,args.stale_threshold,mask,args.eval_batch_size); ta=apply(model,ts,tq,th,args.topk,args.stale_threshold,mask,args.eval_batch_size); vo=evaluate_filtered(va,vq,vf); to=evaluate_filtered(ta,tq,tf); td=evaluate_diagnostics(ta,tq,tf,th,args.stale_threshold); d={'dataset':args.dataset,'variant':v,'seed':args.seed,'topk':args.topk,'feature_mask':mask,'disable_relation_context':disrel,'valid_overall':vo,'test_overall':to,'test_diagnostics':td,'train_log':log}; save_json(d,out/f'{v}_results.json'); torch.save(model.state_dict(),out/f'{v}_adapter.pt');
        if args.save_adjusted: np.savez_compressed(out/f'{v}_test_adjusted_scores.npz',scores=ta,triples=tq,entity=np.asarray(['object']))
        rows.append({'variant':v,'MRR':to['MRR'],'Hits@1':to['Hits@1'],'Hits@3':to['Hits@3'],'Hits@10':to['Hits@10'],'Repeat_H1':td['repeat']['Hits@1'],'Near_H1':td['near_repeat']['Hits@1'],'Novel_H1':td['novel']['Hits@1'],'StaleTop1':td['StaleTop1']}); print(v,to,td['StaleTop1'])
    import pandas as pd; pd.DataFrame(rows).to_csv(out/'ablation_summary.csv',index=False)
if __name__=='__main__': main()
