#!/usr/bin/env python3
import argparse, glob, json
from pathlib import Path
import pandas as pd
ap=argparse.ArgumentParser(); ap.add_argument('--glob',action='append',required=True); ap.add_argument('--out-csv',default='results/all_json_metrics.csv'); a=ap.parse_args(); files=[]
for pat in a.glob: files+=glob.glob(pat,recursive=True)
rows=[]
for f in sorted(set(files)):
    try: d=json.load(open(f,encoding='utf-8'))
    except: continue
    m=d.get('test_overall') or d.get('overall') or d.get('adjusted_metrics') or {}; diag=d.get('test_diagnostics') or {}
    rows.append({'source':f,'dataset':d.get('dataset',''),'method':d.get('method',d.get('variant',d.get('mode',Path(f).parent.name))),'seed':d.get('seed',''),'MRR':m.get('MRR',''),'Hits@1':m.get('Hits@1',''),'Hits@3':m.get('Hits@3',''),'Hits@10':m.get('Hits@10',''),'StaleTop1':diag.get('StaleTop1','')})
Path(a.out_csv).parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(a.out_csv,index=False); print(f'{len(rows)} rows -> {a.out_csv}')
