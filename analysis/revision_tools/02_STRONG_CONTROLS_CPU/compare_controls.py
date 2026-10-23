#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from revision_common import write_csv
ap=argparse.ArgumentParser(); ap.add_argument('--json',action='append',required=True); ap.add_argument('--out-csv',default='results/control_comparison.csv'); a=ap.parse_args(); rows=[]
for f in a.json:
    d=json.load(open(f,encoding='utf-8')); m=d.get('test_overall') or d.get('overall') or {}; rows.append({'source':f,'dataset':d.get('dataset',''),'method':d.get('method',Path(f).parent.name),'MRR':m.get('MRR',''),'Hits@1':m.get('Hits@1',''),'Hits@3':m.get('Hits@3',''),'Hits@10':m.get('Hits@10','')})
write_csv(a.out_csv,rows); print(Path(a.out_csv).resolve())
