#!/usr/bin/env python3
import argparse
from revision_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--base-dump',required=True); ap.add_argument('--adjusted-dump'); ap.add_argument('--split',choices=['valid','test'],default='test'); ap.add_argument('--stale-threshold',type=int,default=10); ap.add_argument('--out-csv',default='results/diagnostic_groups.csv'); ap.add_argument('--out-json',default='results/diagnostic_audit.json'); a=ap.parse_args()
ne,nr,sp,h,filt=context(a.data_root,a.dataset,a.split); bs,tr,e=load_score_dump(a.base_dump); q=canonicalize_queries(tr,e,nr); br=filtered_ranks(bs,q,filt); fmap=build_filter_map(filt)
hs=ar=None
if a.adjusted_dump:
    hs,htr,he=load_score_dump(a.adjusted_dump); hq=canonicalize_queries(htr,he,nr); align_dumps(q,hq); ar=filtered_ranks(hs,q,filt)
buckets=np.asarray([novelty_bucket(*map(int,x[:4]),h) for x in q],dtype=object)
rows=[]
for b in ('repeat','near_repeat','novel'):
    m=buckets==b; row={'group':b,'count':int(m.sum()),'fraction':float(m.mean()),'base_MRR':metrics_from_ranks(br[m])['MRR'],'base_Hits@1':metrics_from_ranks(br[m])['Hits@1']}
    if ar is not None: row.update({'adjusted_MRR':metrics_from_ranks(ar[m])['MRR'],'adjusted_Hits@1':metrics_from_ranks(ar[m])['Hits@1']})
    rows.append(row)

def stale_top1(scores):
    n=0
    for row,x in zip(scores,q):
        s,r,o,t=map(int,x[:4]); y=row.copy()
        for c in fmap.get((s,r,t),()):
            if c!=o and 0<=c<len(y): y[c]=-np.inf
        c=int(np.argmax(y))
        if c!=o and stale_candidate(s,r,c,t,h,a.stale_threshold): n+=1
    return n/len(q) if len(q) else 0.0
base_all=metrics_from_ranks(br); out={'groups_are_exhaustive':bool(sum(r['count'] for r in rows)==len(q)),'groups_are_mutually_exclusive_by_construction':True,'num_queries':len(q),'base_overall':base_all,'base_StaleTop1':stale_top1(bs)}
weighted=sum(r['fraction']*r['base_Hits@1'] for r in rows); out['base_weighted_group_Hits@1']=weighted; out['base_overall_minus_weighted_Hits@1']=base_all['Hits@1']-weighted
if ar is not None:
    adj_all=metrics_from_ranks(ar); weighted2=sum(r['fraction']*r['adjusted_Hits@1'] for r in rows); out.update({'adjusted_overall':adj_all,'adjusted_StaleTop1':stale_top1(hs),'adjusted_weighted_group_Hits@1':weighted2,'adjusted_overall_minus_weighted_Hits@1':adj_all['Hits@1']-weighted2})
write_csv(a.out_csv,rows); save_json(out,a.out_json); print(out)
