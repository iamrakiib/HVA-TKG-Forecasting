#!/usr/bin/env python3
import argparse
from revision_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--base-dump',required=True); ap.add_argument('--adjusted-dump',required=True); ap.add_argument('--split',choices=['valid','test'],default='test'); ap.add_argument('--topk',type=int,default=100); ap.add_argument('--stale-threshold',type=int,default=10); ap.add_argument('--max-per-type',type=int,default=5); ap.add_argument('--out-csv',default='results/case_studies.csv'); a=ap.parse_args()
ne,nr,sp,h,filt=context(a.data_root,a.dataset,a.split); bs,tr,e=load_score_dump(a.base_dump); hs,htr,he=load_score_dump(a.adjusted_dump); q=canonicalize_queries(tr,e,nr); hq=canonicalize_queries(htr,he,nr); align_dumps(q,hq); rb=filtered_ranks(bs,q,filt); rh=filtered_ranks(hs,q,filt); fmap=build_filter_map(filt); ids=topk_indices(bs,a.topk); gold=q[:,2].astype(int); inside=np.any(ids==gold[:,None],axis=1)
rows=[]; counts={}
for i,(rowb,rowh,x) in enumerate(zip(bs,hs,q)):
    s,r,o,t=map(int,x[:4]); fb=rowb.copy(); fh=rowh.copy()
    for c in fmap.get((s,r,t),()):
        if c!=o: fb[c]=fh[c]=-np.inf
    btop=int(np.argmax(fb)); htop=int(np.argmax(fh)); typ=None
    if rb[i]>1 and rh[i]==1: typ='wrong_to_correct'
    elif rb[i]==1 and rh[i]>1: typ='correct_to_wrong'
    elif rb[i]>1 and stale_candidate(s,r,btop,t,h,a.stale_threshold) and htop!=btop: typ='stale_top1_demoted'
    elif not inside[i]: typ='gold_outside_topk'
    if typ is None or counts.get(typ,0)>=a.max_per_type: continue
    times=h['sr'].get((s,r),{}).get(btop,[]); lt=last_time_before(times,t); feat=exact_features(times,t,a.stale_threshold)
    rows.append({'case_type':typ,'query_index':i,'s':s,'r':r,'gold_o':o,'t':t,'base_top1':btop,'adjusted_top1':htop,'base_gold_rank':int(rb[i]),'adjusted_gold_rank':int(rh[i]),'gold_inside_base_topk':bool(inside[i]),'base_top1_last_exact_time':lt if lt is not None else '','base_top1_gap':'' if lt is None else int(t)-lt,'base_top1_occurrence':feat[0],'base_top1_recency':feat[1],'base_top1_logfreq':feat[2],'base_top1_stale':feat[3]}); counts[typ]=counts.get(typ,0)+1
write_csv(a.out_csv,rows); print(f'wrote {len(rows)} cases -> {a.out_csv}')
