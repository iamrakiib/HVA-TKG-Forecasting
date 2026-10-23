#!/usr/bin/env python3
import argparse, os
import numpy as np, pandas as pd
from hva_core.history_utils import *
from hva_core.metrics import evaluate_filtered
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--base-dump',required=True); ap.add_argument('--exact-dump',required=True); ap.add_argument('--dual-dump',required=True); ap.add_argument('--out-csv',default='results/exact_vs_dual_slices.csv'); a=ap.parse_args(); data=os.path.join(a.data_root,a.dataset); ne,nr=read_stat(data); sp=read_split_arrays(data); bs,btr,be=load_score_dump(a.base_dump); es,etr,ee=load_score_dump(a.exact_dump); ds,dtr,de=load_score_dump(a.dual_dump); q=canonicalize_queries(btr,be,nr); eq=canonicalize_queries(etr,ee,nr); dq=canonicalize_queries(dtr,de,nr); assert np.array_equal(q,eq) and np.array_equal(q,dq); hist=build_histories(augment_with_inverse([tuple(x) for x in np.concatenate([sp['train'],sp['valid']],0)],nr)); filt=augment_with_inverse([tuple(x) for x in np.concatenate([sp['train'],sp['valid'],sp['test']],0)],nr)
labels=[]
for s,r,o,t in q:
    sr=last_time_before(hist['sr'].get((int(s),int(r)),{}).get(int(o),[]),int(t)); so=last_time_before(hist['so'].get(int(s),{}).get(int(o),[]),int(t)); ro=last_time_before(hist['ro'].get(int(r),{}).get(int(o),[]),int(t)); labels.append('exact_history' if sr is not None else ('near_history_only' if (so is not None or ro is not None) else 'novel'))
labels=np.asarray(labels); rows=[]
for lab in ('exact_history','near_history_only','novel'):
    m=labels==lab; rows.append({'slice':lab,'count':int(m.sum()),'base_MRR':evaluate_filtered(bs[m],q[m],filt)['MRR'],'exact_MRR':evaluate_filtered(es[m],q[m],filt)['MRR'],'dual_MRR':evaluate_filtered(ds[m],q[m],filt)['MRR']})
Path=os.path; os.makedirs(os.path.dirname(a.out_csv) or '.',exist_ok=True); pd.DataFrame(rows).to_csv(a.out_csv,index=False); print(pd.DataFrame(rows))
