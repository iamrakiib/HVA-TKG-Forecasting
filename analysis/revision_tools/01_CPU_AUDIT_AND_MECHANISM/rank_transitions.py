#!/usr/bin/env python3
import argparse
from revision_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--base-dump',required=True); ap.add_argument('--adjusted-dump',required=True); ap.add_argument('--split',choices=['valid','test'],default='test'); ap.add_argument('--out-csv',default='results/rank_transitions.csv'); ap.add_argument('--out-json',default='results/rank_transition_summary.json'); a=ap.parse_args()
ne,nr,sp,h,filt=context(a.data_root,a.dataset,a.split); bs,tr,e=load_score_dump(a.base_dump); hs,htr,he=load_score_dump(a.adjusted_dump); q=canonicalize_queries(tr,e,nr); hq=canonicalize_queries(htr,he,nr); align_dumps(q,hq); rb=filtered_ranks(bs,q,filt); rh=filtered_ranks(hs,q,filt)
bc=rb==1; hc=rh==1; cats={'wrong_to_correct':(~bc)&hc,'correct_to_wrong':bc&(~hc),'correct_to_correct':bc&hc,'wrong_to_wrong':(~bc)&(~hc)}
rows=[{'transition':k,'count':int(v.sum()),'fraction':float(v.mean())} for k,v in cats.items()]; write_csv(a.out_csv,rows)
out={'num_queries':len(q),'rank_improved':int((rh<rb).sum()),'rank_unchanged':int((rh==rb).sum()),'rank_worsened':int((rh>rb).sum()),'net_top1_corrections':int(cats['wrong_to_correct'].sum()-cats['correct_to_wrong'].sum()),'base_metrics':metrics_from_ranks(rb),'adjusted_metrics':metrics_from_ranks(rh)}; save_json(out,a.out_json); print(out)
