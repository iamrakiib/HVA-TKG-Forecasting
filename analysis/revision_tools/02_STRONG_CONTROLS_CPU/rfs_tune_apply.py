#!/usr/bin/env python3
import argparse, itertools, json
from pathlib import Path
from revision_common import *

def feats_topk(scores,q,h,k,thr):
    ids=topk_indices(scores,k); F=np.zeros((len(q),ids.shape[1],4),np.float32)
    sr=h['sr']
    for i,(s,r,_o,t) in enumerate(q):
        m=sr.get((int(s),int(r)),{})
        for j,c in enumerate(ids[i]): F[i,j]=exact_features(m.get(int(c),[]),int(t),thr)
    return ids,F

def apply(scores,ids,F,w):
    out=scores.copy(); corr=F@np.asarray([w[0],w[1],w[2],-w[3]],dtype=np.float32)
    rr=np.arange(len(out))[:,None]; out[rr,ids]=out[rr,ids]+corr; return out

def parse(s): return [float(x) for x in s.split(',') if x.strip()]
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--valid-dump',required=True); ap.add_argument('--test-dump',required=True); ap.add_argument('--topk',type=int,default=100); ap.add_argument('--stale-threshold',type=int,default=10); ap.add_argument('--occ-grid',default='0,0.05'); ap.add_argument('--rec-grid',default='0,0.05,0.10'); ap.add_argument('--freq-grid',default='0,0.025,0.05'); ap.add_argument('--stale-grid',default='0,0.05,0.10'); ap.add_argument('--out-dir',default='results/rfs'); a=ap.parse_args()
# valid context
_,nr,sp,vh,vf=context(a.data_root,a.dataset,'valid'); vs,vtr,ve=load_score_dump(a.valid_dump); vq=canonicalize_queries(vtr,ve,nr)
_,_,_,th,tf=context(a.data_root,a.dataset,'test'); ts,ttr,te=load_score_dump(a.test_dump); tq=canonicalize_queries(ttr,te,nr)
vid,vF=feats_topk(vs,vq,vh,a.topk,a.stale_threshold); tid,tF=feats_topk(ts,tq,th,a.topk,a.stale_threshold)
best=None; bestm=-1; trials=[]
for w in itertools.product(parse(a.occ_grid),parse(a.rec_grid),parse(a.freq_grid),parse(a.stale_grid)):
    adj=apply(vs,vid,vF,w); m=evaluate_scores(adj,vq,vf); trials.append({'w_occ':w[0],'w_rec':w[1],'w_freq':w[2],'w_stale':w[3],'valid_MRR':m['MRR']})
    if m['MRR']>bestm: bestm=m['MRR']; best=w
vadj=apply(vs,vid,vF,best); tadj=apply(ts,tid,tF,best); outdir=Path(a.out_dir); outdir.mkdir(parents=True,exist_ok=True)
res={'dataset':a.dataset,'method':'RFS_same_exact_features','topk':a.topk,'stale_threshold':a.stale_threshold,'selected_weights':{'occurrence':best[0],'recency':best[1],'log_frequency':best[2],'stale_penalty':best[3]},'valid_overall':evaluate_scores(vadj,vq,vf),'test_overall':evaluate_scores(tadj,tq,tf),'validation_trials':trials,'note':'Parameters selected on VALID only; TEST used once after selection.'}; save_json(res,outdir/'rfs_results.json'); np.savez_compressed(outdir/'valid_rfs_scores.npz',scores=vadj,triples=vq,entity=np.asarray(['object'])); np.savez_compressed(outdir/'test_rfs_scores.npz',scores=tadj,triples=tq,entity=np.asarray(['object'])); print(json.dumps(res,indent=2))
