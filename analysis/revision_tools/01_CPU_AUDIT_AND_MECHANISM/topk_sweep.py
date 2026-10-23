#!/usr/bin/env python3
import argparse
from revision_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--base-dump',required=True); ap.add_argument('--adjusted-dump'); ap.add_argument('--split',choices=['valid','test'],default='test'); ap.add_argument('--k-list',default='20,50,100,200,500'); ap.add_argument('--out-csv',default='results/topk_sweep.csv'); a=ap.parse_args()
ne,nr,sp,h,filt=context(a.data_root,a.dataset,a.split); bs,tr,e=load_score_dump(a.base_dump); q=canonicalize_queries(tr,e,nr); br=filtered_ranks(bs,q,filt)
ars=None
if a.adjusted_dump:
    hs,htr,he=load_score_dump(a.adjusted_dump); hq=canonicalize_queries(htr,he,nr); align_dumps(q,hq); ar=filtered_ranks(hs,q,filt)
rows=[]; gold=q[:,2].astype(int)
for k in [int(x) for x in a.k_list.split(',') if x.strip()]:
    ids=topk_indices(bs,k); inside=np.any(ids==gold[:,None],axis=1)
    row={'K':k,'num_queries':len(q),'gold_recall_at_K':float(inside.mean()),'gold_inside':int(inside.sum()),'gold_outside':int((~inside).sum()),'base_MRR_inside':metrics_from_ranks(br[inside])['MRR'],'base_MRR_outside':metrics_from_ranks(br[~inside])['MRR']}
    if a.adjusted_dump:
        row.update({'adjusted_MRR_inside_on_base_K_subset':metrics_from_ranks(ar[inside])['MRR'],'adjusted_MRR_outside_on_base_K_subset':metrics_from_ranks(ar[~inside])['MRR']})
    rows.append(row)
write_csv(a.out_csv,rows); print(Path(a.out_csv).resolve())
