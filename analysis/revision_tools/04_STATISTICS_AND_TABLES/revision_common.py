#!/usr/bin/env python3
from __future__ import annotations
import bisect, json, os
from pathlib import Path
from typing import Dict, Tuple, Sequence
import numpy as np

SCORE_KEYS = ('scores','base_scores','logits','predictions')
TRIPLE_KEYS = ('triples','queries','quads','facts','trips')

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)

def save_json(obj, path):
    ensure_dir(Path(path).parent)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,sort_keys=True)

def load_score_dump(path):
    obj = np.load(path, allow_pickle=True)
    sk = next((k for k in SCORE_KEYS if k in obj), None)
    tk = next((k for k in TRIPLE_KEYS if k in obj), None)
    if sk is None or tk is None:
        raise KeyError(f"{path}: need score key from {SCORE_KEYS} and triple key from {TRIPLE_KEYS}; got {list(obj.files)}")
    scores = np.asarray(obj[sk], dtype=np.float32)
    triples = np.asarray(obj[tk], dtype=np.int64)
    if scores.ndim != 2 or triples.ndim != 2 or triples.shape[1] < 4 or len(scores)!=len(triples):
        raise ValueError(f"Bad dump shapes scores={scores.shape}, triples={triples.shape}")
    entity='object'
    if 'entity' in obj:
        arr=np.asarray(obj['entity']).reshape(-1)
        if len(arr): entity=str(arr[0])
    return scores, triples[:,:4], entity

def read_stat(data_dir):
    p=Path(data_dir)/'stat.txt'
    parts=p.read_text(encoding='utf-8').strip().split()
    if len(parts)<2: raise ValueError(f'Bad stat.txt: {p}')
    return int(parts[0]), int(parts[1])

def read_split(path):
    rows=[]
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            z=line.strip().split()
            if len(z)>=4: rows.append(tuple(map(int,z[:4])))
    return np.asarray(rows,dtype=np.int64)

def read_splits(data_dir):
    return {s:read_split(Path(data_dir)/f'{s}.txt') for s in ('train','valid','test')}

def augment_with_inverse(arr, num_rels):
    arr=np.asarray(arr,dtype=np.int64)
    out=[]
    for s,r,o,t in arr:
        out.append((int(s),int(r),int(o),int(t)))
        out.append((int(o),int(r)+int(num_rels),int(s),int(t)))
    return out

def canonicalize_queries(triples, entity, num_rels):
    arr=np.asarray(triples,dtype=np.int64).copy()
    e=str(entity).lower()
    if e in ('object','tail','o'): return arr
    if e in ('subject','head','s'):
        src=arr.copy(); arr[:,0]=src[:,2]; arr[:,1]=src[:,1]+int(num_rels); arr[:,2]=src[:,0]; return arr
    raise ValueError(f'Unknown entity branch: {entity}')

def build_histories(triples):
    sr={}; so={}; ro={}
    for s,r,o,t in triples:
        sr.setdefault((int(s),int(r)),{}).setdefault(int(o),[]).append(int(t))
        so.setdefault(int(s),{}).setdefault(int(o),[]).append(int(t))
        ro.setdefault(int(r),{}).setdefault(int(o),[]).append(int(t))
    for H in (sr,so,ro):
        for cmap in H.values():
            for times in cmap.values(): times.sort()
    return {'sr':sr,'so':so,'ro':ro}

def last_time_before(times,t):
    i=bisect.bisect_left(times,int(t))-1
    return None if i<0 else int(times[i])

def freq_before(times,t): return int(bisect.bisect_left(times,int(t)))

def exact_features(times,t,stale_threshold=10):
    lt=last_time_before(times,int(t))
    if lt is None: return (0.0,0.0,0.0,0.0)
    gap=max(0,int(t)-lt); freq=freq_before(times,int(t))
    return (1.0, 1.0/(1.0+float(gap)), float(np.log1p(freq)), 1.0 if gap>int(stale_threshold) else 0.0)

def build_filter_map(triples):
    out={}
    for s,r,o,t in triples: out.setdefault((int(s),int(r),int(t)),set()).add(int(o))
    return out

def filtered_rank(row,q,fmap):
    s,r,o,t=map(int,q[:4]); x=np.asarray(row,dtype=np.float32).copy()
    for c in fmap.get((s,r,t),()):
        if c!=o and 0<=c<len(x): x[c]=-np.inf
    gs=x[o]
    return int(np.sum(x>gs)+1)

def filtered_ranks(scores,queries,filter_triples):
    fmap=build_filter_map(filter_triples)
    return np.asarray([filtered_rank(row,q,fmap) for row,q in zip(scores,queries)],dtype=np.int64)

def metrics_from_ranks(ranks):
    r=np.asarray(ranks,dtype=np.float64)
    if len(r)==0: return {'MRR':0.0,'Hits@1':0.0,'Hits@3':0.0,'Hits@10':0.0,'count':0}
    return {'MRR':float(np.mean(1.0/r)),'Hits@1':float(np.mean(r<=1)),'Hits@3':float(np.mean(r<=3)),'Hits@10':float(np.mean(r<=10)),'count':int(len(r))}

def evaluate_scores(scores,queries,filter_triples): return metrics_from_ranks(filtered_ranks(scores,queries,filter_triples))

def topk_indices(scores,k):
    """Use exactly the same torch.topk candidate selection as HVA.

    Important for tied score vectors: NumPy argpartition can select a
    different top-K set. Score tensors are cast to float32 like HVA.
    """
    import torch
    score_t=torch.as_tensor(np.asarray(scores),dtype=torch.float32)
    if score_t.ndim!=2 or int(k)<=0:
        raise ValueError('Expected scores [N,E] and positive K')
    return torch.topk(score_t.detach(),k=min(int(k),int(score_t.shape[1])),dim=1).indices.cpu().numpy()


def novelty_bucket(s,r,o,t,h):
    if last_time_before(h['sr'].get((int(s),int(r)),{}).get(int(o),[]),t) is not None: return 'repeat'
    a=last_time_before(h['so'].get(int(s),{}).get(int(o),[]),t) is not None
    b=last_time_before(h['ro'].get(int(r),{}).get(int(o),[]),t) is not None
    return 'near_repeat' if (a or b) else 'novel'

def stale_candidate(s,r,c,t,h,threshold=10):
    lt=last_time_before(h['sr'].get((int(s),int(r)),{}).get(int(c),[]),t)
    return False if lt is None else (int(t)-lt)>int(threshold)

def context(data_root,dataset,split):
    data_dir=Path(data_root)/dataset
    ne,nr=read_stat(data_dir); sp=read_splits(data_dir)
    if split=='valid':
        hist=augment_with_inverse(sp['train'],nr)
        filt=augment_with_inverse(np.concatenate([sp['train'],sp['valid']],axis=0),nr)
    elif split=='test':
        hist=augment_with_inverse(np.concatenate([sp['train'],sp['valid']],axis=0),nr)
        filt=augment_with_inverse(np.concatenate([sp['train'],sp['valid'],sp['test']],axis=0),nr)
    else: raise ValueError('split must be valid or test')
    return ne,nr,sp,build_histories(hist),filt

def align_dumps(aq,bq):
    if len(aq)!=len(bq) or not np.array_equal(np.asarray(aq)[:,:4],np.asarray(bq)[:,:4]):
        raise ValueError('Score dumps do not contain identical queries/order')

def write_csv(path, rows):
    import pandas as pd
    ensure_dir(Path(path).parent); pd.DataFrame(rows).to_csv(path,index=False)
