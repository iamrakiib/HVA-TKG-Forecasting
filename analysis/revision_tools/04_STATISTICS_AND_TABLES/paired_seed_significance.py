#!/usr/bin/env python3
import argparse, math
import numpy as np, pandas as pd
from scipy.stats import ttest_rel, wilcoxon, t
ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--baseline',default='baseline'); ap.add_argument('--target',default='HVA exact-only'); ap.add_argument('--metric',default='MRR'); ap.add_argument('--out-csv',default='results/significance.csv'); a=ap.parse_args(); d=pd.read_csv(a.csv); rows=[]
for (ds,bb),g in d.groupby(['dataset','backbone']):
    A=g[g.method==a.baseline].set_index('seed')[a.metric]; B=g[g.method==a.target].set_index('seed')[a.metric]; seeds=sorted(set(A.index)&set(B.index)); x=A.loc[seeds].astype(float).values; y=B.loc[seeds].astype(float).values; diff=y-x; n=len(diff)
    if n<2: continue
    tt=ttest_rel(y,x); sd=diff.std(ddof=1); se=sd/math.sqrt(n); crit=t.ppf(.975,n-1); ci=(diff.mean()-crit*se,diff.mean()+crit*se); dz=diff.mean()/sd if sd>0 else float('inf');
    try: wp=wilcoxon(y,x,alternative='two-sided',mode='auto').pvalue
    except Exception: wp=float('nan')
    rows.append({'dataset':ds,'backbone':bb,'n_seeds':n,'baseline_mean':x.mean(),'target_mean':y.mean(),'mean_delta':diff.mean(),'delta_std':sd,'delta_CI95_low':ci[0],'delta_CI95_high':ci[1],'paired_t_p':tt.pvalue,'wilcoxon_p':wp,'cohen_dz':dz,'positive_seed_fraction':float(np.mean(diff>0))})
from pathlib import Path; Path(a.out_csv).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out_csv,index=False); print(pd.DataFrame(rows).to_string(index=False))
