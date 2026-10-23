#!/usr/bin/env python3
import argparse
from revision_common import *
ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--data-root',required=True); ap.add_argument('--base-dump',required=True); ap.add_argument('--adjusted-dump'); ap.add_argument('--split',choices=['valid','test'],default='test'); ap.add_argument('--out-csv',default='results/history_age_mechanism.csv'); a=ap.parse_args()
ne,nr,sp,h,filt=context(a.data_root,a.dataset,a.split); bs,tr,e=load_score_dump(a.base_dump); q=canonicalize_queries(tr,e,nr); br=filtered_ranks(bs,q,filt)
ar=None
if a.adjusted_dump:
    hs,htr,he=load_score_dump(a.adjusted_dump); hq=canonicalize_queries(htr,he,nr); align_dumps(q,hq); ar=filtered_ranks(hs,q,filt)
labels=[]
for s,r,o,t in q:
    times=h['sr'].get((int(s),int(r)),{}).get(int(o),[]); lt=last_time_before(times,int(t))
    if lt is None: lab='no_exact_history'
    else:
        g=int(t)-lt
        lab='gap_1' if g<=1 else ('gap_2_3' if g<=3 else ('gap_4_10' if g<=10 else ('gap_11_30' if g<=30 else 'gap_31_plus')))
    labels.append(lab)
labels=np.asarray(labels,dtype=object); rows=[]
for lab in ('gap_1','gap_2_3','gap_4_10','gap_11_30','gap_31_plus','no_exact_history'):
    m=labels==lab; bm=metrics_from_ranks(br[m]); row={'age_bucket':lab,'count':int(m.sum()),'fraction':float(m.mean()),'base_MRR':bm['MRR'],'base_Hits@1':bm['Hits@1']}
    if ar is not None:
        hm=metrics_from_ranks(ar[m]); row.update({'adjusted_MRR':hm['MRR'],'adjusted_Hits@1':hm['Hits@1'],'delta_MRR':hm['MRR']-bm['MRR'],'delta_Hits@1':hm['Hits@1']-bm['Hits@1']})
    rows.append(row)
write_csv(a.out_csv,rows); print(Path(a.out_csv).resolve())
