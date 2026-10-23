#!/usr/bin/env python3
"""Run the manuscript's later diagnostics on one genuine score-dump population.

Produces new output files only after user invocation. No results are supplied.
For a separate 10k diagnostic sample, pass that sample's actual score dumps;
this script never manufactures or resamples query instances to fit a count.
"""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SCRIPTS=['audit_inputs.py','topk_sweep.py','diagnostic_group_audit.py',
         'history_age_mechanism.py','rank_transitions.py','extract_case_studies.py']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--data-root',required=True)
    p.add_argument('--base-dump',required=True);p.add_argument('--adjusted-dump',required=True)
    p.add_argument('--split',choices=['valid','test'],default='test')
    p.add_argument('--out-root',type=Path,required=True)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if ROOT in a.out_root.resolve().parents or a.out_root.resolve()==ROOT:raise ValueError('Output root must be outside repository')
    cwd=ROOT/'analysis/revision_tools/01_CPU_AUDIT_AND_MECHANISM'
    for name in SCRIPTS:
        cmd=[sys.executable,name,'--dataset',a.dataset,'--data-root',str(Path(a.data_root).resolve()),
             '--split',a.split]
        if name=='audit_inputs.py':
            cmd+=['--dump',str(Path(a.base_dump).resolve()),'--out',str(a.out_root/'audit_inputs.json')]
        else:
            cmd+=['--base-dump',str(Path(a.base_dump).resolve())]
            if name!='topk_sweep.py' or a.adjusted_dump:
                cmd+=['--adjusted-dump',str(Path(a.adjusted_dump).resolve())]
            filearg={'topk_sweep.py':('--out-csv','topk_sweep.csv'),
                     'diagnostic_group_audit.py':('--out-csv','diagnostic_groups.csv'),
                     'history_age_mechanism.py':('--out-csv','history_age.csv'),
                     'rank_transitions.py':('--out-csv','rank_transitions.csv'),
                     'extract_case_studies.py':('--out-csv','case_studies.csv')}
            arg,fname=filearg[name];cmd+=[arg,str(a.out_root/fname)]
            if name=='diagnostic_group_audit.py':cmd+=['--out-json',str(a.out_root/'diagnostic_audit.json')]
            if name=='rank_transitions.py':cmd+=['--out-json',str(a.out_root/'rank_transitions.json')]
        print('$',' '.join(map(str,cmd)),flush=True)
        if a.execute:subprocess.run(cmd,cwd=cwd,check=True)
if __name__=='__main__':main()
