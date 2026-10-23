#!/usr/bin/env python3
"""Single-population arithmetic audit: Recall@K, inside/outside MRR and overall.

This script never supplies a target percentage: it computes everything from
actual original-query score arrays and prints sufficient counts to make all
rounding decisions reproducible.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'analysis/revision_tools/01_CPU_AUDIT_AND_MECHANISM'))
from revision_common import *

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True)
    ap.add_argument('--data-root',required=True);ap.add_argument('--base-dump',required=True)
    ap.add_argument('--adjusted-dump',required=True);ap.add_argument('--k',type=int,default=100)
    a=ap.parse_args()
    _,nrel,sp,h,filt=context(a.data_root,a.dataset,'test')
    base,orig,branch=load_score_dump(a.base_dump)
    adj,adapt,branch2=load_score_dump(a.adjusted_dump)
    if branch!='object' or branch2!='object':raise ValueError('Expected object prediction dumps')
    align_dumps(orig,adapt);q=canonicalize_queries(orig,branch,nrel)
    gold=q[:,2];top=topk_indices(base,a.k);inside=np.any(top==gold[:,None],axis=1)
    br=filtered_ranks(base,q,filt);ar=filtered_ranks(adj,q,filt)
    p=float(inside.mean()); print('N=',len(q),'inside=',int(inside.sum()),'outside=',int((~inside).sum()))
    print('Recall@K exact=',p,'percentage=',f'{p:.8%}')
    for label,ranks in [('backbone',br),('HVA',ar)]:
        all_m=metrics_from_ranks(ranks)['MRR'];in_m=metrics_from_ranks(ranks[inside])['MRR'];out_m=metrics_from_ranks(ranks[~inside])['MRR']
        weighted=p*in_m+(1-p)*out_m
        assert np.isclose(all_m,weighted,atol=1e-12), (all_m,weighted)
        print(label,{'overall_mrr':all_m,'inside_mrr':in_m,'outside_mrr':out_m,'weighted_mrr':weighted})
if __name__=='__main__':main()
