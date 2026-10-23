#!/usr/bin/env python3
"""Standalone *HRI-inspired* recurrence control; no backbone HVA adapter.

This source is distinct from the third-party HRI project. Its validation grid
and test scores are computed from real dataset files at execution time. It
cannot by itself establish numerical equivalence to published HRI variants or
previous five-seed scores, and it does not bundle third-party HRI source.
"""
import argparse,sys,itertools,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backbone_integrations/tirgn'))
from hva.run_hri_baseline import build_hri_scores
from hva.history_utils import read_stat,read_split_arrays,augment_with_inverse,build_histories
from hva.metrics import evaluate_filtered
from hri_tie_audit import evaluate_hri_ties
from run_experiment_matrix import assert_ordinal_timestamps

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset',required=True);ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--alpha-freq',default='0.0,0.5,1.0')
    ap.add_argument('--alpha-recency',default='0.5,1.0,2.0')
    ap.add_argument('--lambda-decay',default='0.01,0.03,0.1')
    ap.add_argument('--allow-unverified-time-units',action='store_true',
                    help='Use archived raw-time values despite paper snapshot-index definition')
    ap.add_argument('--out-json',required=True,type=Path)
    ap.add_argument('--tie-policy',choices=('legacy_strict','expected_uniform'),
                    default='legacy_strict', help='Main reported metric; default follows manuscript strict-> ranking')
    a=ap.parse_args();data=a.data_root/a.dataset
    if not a.allow_unverified_time_units:
        assert_ordinal_timestamps({'data_root':str(a.data_root),'dataset':a.dataset})
    num_e,num_r=read_stat(data);s=read_split_arrays(data)
    valid_history=build_histories(augment_with_inverse(s['train'],num_r))
    test_history=build_histories(augment_with_inverse(np.concatenate((s['train'],s['valid'])),num_r))
    valid_filter=augment_with_inverse(np.concatenate((s['train'],s['valid'])),num_r)
    test_filter=augment_with_inverse(np.concatenate((s['train'],s['valid'],s['test'])),num_r)
    def grid(v):return [float(i) for i in v.split(',')]
    configs=itertools.product(grid(a.alpha_freq),grid(a.alpha_recency),grid(a.lambda_decay))
    best=None;best_val=-np.inf
    for af,ar,lam in configs:
        vscore=build_hri_scores(s['valid'],num_e,valid_history,af,ar,lam)
        m=evaluate_hri_ties(vscore,s['valid'],valid_filter,policy=a.tie_policy)['MRR']
        if m>best_val:best_val=m;best=(af,ar,lam)
    ts=build_hri_scores(s['test'],num_e,test_history,*best)
    out={'dataset':a.dataset,'model':'standalone HRI-inspired independent recurrence control',
         'validation_selected':{'alpha_frequency':best[0],'alpha_recency':best[1],'lambda_decay':best[2]},
         'validation_MRR':best_val,
         'test':evaluate_hri_ties(ts,s['test'],test_filter,policy=a.tie_policy),
         'tie_sensitivity':{
             'legacy_strict':evaluate_hri_ties(ts,s['test'],test_filter,policy='legacy_strict'),
             'expected_uniform':evaluate_hri_ties(ts,s['test'],test_filter,policy='expected_uniform')
         },
         'warning':'legacy_strict counts only scores > gold and can give rank 1 to all-zero ties; expected_uniform is sensitivity analysis, not a reconstruction of published HRI settings',
         'history':'validation=train only; test=train+valid only, strict t_previous < t_query'}
    if ROOT in a.out_json.resolve().parents:raise ValueError('Write generated outputs outside repository')
    a.out_json.parent.mkdir(parents=True,exist_ok=True)
    a.out_json.write_text(json.dumps(out,indent=2),encoding='utf8')
    print('Computed and saved standalone recurrence evaluation:',a.out_json)
    if out['test']['flat_score_rows']:
        print('WARNING: flat HRI score vectors detected:',out['test']['flat_score_rows'],
              'The manuscript strict-greater tie policy can overstate their ranks.',file=sys.stderr)
if __name__=='__main__':main()
