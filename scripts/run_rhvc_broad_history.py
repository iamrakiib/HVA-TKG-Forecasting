#!/usr/bin/env python3
"""RHVC broader-history score correction matching its ROLE in the HVA manuscript.

This is a new protocol-explicit implementation. The surviving original archived
RHVC scripts were exact-only, so they did NOT implement the manuscript's broader
history claim. This runner incorporates exact (s,r,c), subject-candidate
(s,*,c), relation-candidate (*,r,c) histories. It does not reconstruct the
historical RHVC configuration without the original prototype settings.

The coefficients and decay are REQUIRED from a user-supplied original
configuration or explicit validation grid. No paper metrics are hard-coded.
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BB = ('tirgn','regcn','cygnet','cenet','temp')

@dataclass(frozen=True)
class RHVCParameters:
    gamma: float
    topk: int
    stale_threshold: int
    lambda_decay: float
    exact_occurrence: float
    exact_frequency: float
    exact_recency: float
    exact_stale: float
    subject_occurrence: float
    subject_frequency: float
    subject_recency: float
    subject_stale: float
    relation_occurrence: float
    relation_frequency: float
    relation_recency: float
    relation_stale: float

    @classmethod
    def from_dict(cls, raw: Mapping):
        names={f.name for f in fields(cls)}
        if set(raw) != names:
            raise ValueError(f'RHVC config keys: missing={sorted(names-set(raw))}, extra={sorted(set(raw)-names)}')
        vals=dict(raw)
        vals['topk']=int(vals['topk'])
        vals['stale_threshold']=int(vals['stale_threshold'])
        if vals['topk']<1 or vals['stale_threshold']<0:raise ValueError('topk positive; stale_threshold nonnegative')
        for k in names-{'topk','stale_threshold'}:
            vals[k]=float(vals[k])
            if not np.isfinite(vals[k]):raise ValueError(f'Nonfinite RHVC coefficient {k}')
        return cls(**vals)


def history_signal(times, timestamp:int, tau:int, decay:float)->np.ndarray:
    """Four manuscript quantities [occurrence, log count, decay, staleness].

    All history is pre-indexed; causal cut is enforced by bisect_left.
    Repeated records at the same t count as multiple records, never future data.
    """
    from bisect import bisect_left
    n=bisect_left(times, int(timestamp))
    if n==0:return np.zeros(4,dtype=np.float64)
    age=int(timestamp)-int(times[n-1])
    return np.array([1.,np.log1p(n),np.exp(-decay*age),float(age>tau)],dtype=np.float64)


def apply_rhvc_broad(scores, queries, histories, params:RHVCParameters):
    """Modify only baseline-selected top-K, with no target-dependent insertion.

    The three associations overlap by design; their coefficients are independently
    specified. Scores of all candidates outside top K are byte-identical.
    """
    scores=np.asarray(scores,dtype=np.float32)
    queries=np.asarray(queries,dtype=np.int64)
    if scores.ndim!=2 or queries.ndim!=2 or queries.shape!=(scores.shape[0],4):
        raise ValueError('Expected scores [N,E] and queries [N,4]')
    if not np.isfinite(scores).all():raise ValueError('Invalid backbone scores')
    coeff={
        'sr':np.array([params.exact_occurrence,params.exact_frequency,params.exact_recency,-params.exact_stale],dtype=np.float64),
        'so':np.array([params.subject_occurrence,params.subject_frequency,params.subject_recency,-params.subject_stale],dtype=np.float64),
        'ro':np.array([params.relation_occurrence,params.relation_frequency,params.relation_recency,-params.relation_stale],dtype=np.float64),
    }
    # Same selection primitive as the frozen HVA model.
    from hva.history_utils import score_only_topk
    top=score_only_topk(torch.as_tensor(scores),params.topk).cpu().numpy()
    adjusted=scores.copy()
    sr,so,ro=histories['sr'],histories['so'],histories['ro']
    for i,(sub,rel,_gold,t) in enumerate(queries):
        exact=sr.get((int(sub),int(rel)),{})
        subject=so.get(int(sub),{})
        relation=ro.get(int(rel),{})
        for cand in top[i]:
            c=int(cand)
            delta=(coeff['sr']@history_signal(exact.get(c,()),int(t),params.stale_threshold,params.lambda_decay)
                   +coeff['so']@history_signal(subject.get(c,()),int(t),params.stale_threshold,params.lambda_decay)
                   +coeff['ro']@history_signal(relation.get(c,()),int(t),params.stale_threshold,params.lambda_decay))
            # Same float32 adjustment used by the HVA score-dump overlays.
            adjusted[i,c]=np.float32(adjusted[i,c]+params.gamma*float(delta))
    return adjusted


def context(dataset:str,data_root:Path,valid_dump:Path,test_dump:Path):
    from hva.history_utils import (read_stat,read_split_arrays,load_score_dump,build_histories,
        augment_with_inverse,canonicalize_queries)
    from validate_object_score_dumps import validate
    d=data_root/dataset
    validate(d/'valid.txt',valid_dump,d/'stat.txt')
    validate(d/'test.txt',test_dump,d/'stat.txt')
    ne,nr=read_stat(str(d))
    splits=read_split_arrays(str(d))
    vscore,vq_raw,vbr=load_score_dump(str(valid_dump))
    tscore,tq_raw,tbr=load_score_dump(str(test_dump))
    if vbr!='object' or tbr!='object':raise ValueError('Manuscript reports original-object predictions only')
    vq=canonicalize_queries(vq_raw,vbr,nr)
    tq=canonicalize_queries(tq_raw,tbr,nr)
    hv=build_histories(augment_with_inverse(splits['train'],nr))
    ht=build_histories(augment_with_inverse(np.concatenate([splits['train'],splits['valid']]),nr))
    fv=augment_with_inverse(np.concatenate([splits['train'],splits['valid']]),nr)
    ft=augment_with_inverse(np.concatenate([splits['train'],splits['valid'],splits['test']]),nr)
    return vscore,vq,hv,fv,tscore,tq,ht,ft


def _grid_parameters(file:Path):
    obj=json.loads(file.read_text(encoding='utf-8'))
    names=[f.name for f in fields(RHVCParameters)]
    if set(obj)!=set(names):raise ValueError('Validation grid must specify every RHVC coefficient')
    for name in names:
        if not isinstance(obj[name],list) or not obj[name]:raise ValueError(f'{name} grid must be a nonempty list')
    lens=[len(obj[n]) for n in names]
    if np.prod(lens)>10000:raise ValueError('More than 10000 RHVC configurations; narrow the explicit grid')
    for vals in itertools.product(*(obj[n] for n in names)):
        yield RHVCParameters.from_dict(dict(zip(names,vals)))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backbone',choices=BB,required=True)
    ap.add_argument('--dataset',choices=('ICEWS14','ICEWS18','YAGO'),required=True)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--valid-dump',type=Path,required=True)
    ap.add_argument('--test-dump',type=Path,required=True)
    ap.add_argument('--params-json',type=Path,help='Actual fixed RHVC hyperparameters from run record')
    ap.add_argument('--grid-json',type=Path,help='Actual explicit validation-tuning grid from run record')
    ap.add_argument('--allow-unverified-time-units',action='store_true')
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--save-adjusted-scores',action='store_true')
    a=ap.parse_args(argv)
    if bool(a.params_json)==bool(a.grid_json):ap.error('Provide exactly one of --params-json or --grid-json')
    if ROOT in a.out_dir.resolve().parents or a.out_dir.resolve()==ROOT:
        raise ValueError('Generated results MUST go outside the GitHub source directory')
    sys.path.insert(0,str(ROOT/'backbone_integrations'/a.backbone))
    if not a.allow_unverified_time_units:
        from run_experiment_matrix import assert_ordinal_timestamps
        assert_ordinal_timestamps({'dataset':a.dataset,'data_root':str(a.data_root)})
    from hva.metrics import evaluate_filtered
    vs,vq,vh,vf,ts,tq,th,tf=context(a.dataset,a.data_root,a.valid_dump,a.test_dump)
    if a.params_json:
        params=RHVCParameters.from_dict(json.loads(a.params_json.read_text(encoding='utf-8')))
        selected_by='user_supplied_fixed_configuration'
        candidates=[params]
    else:
        candidates=_grid_parameters(a.grid_json)
        selected_by='validation_MRR_grid_search'
    best,best_val=None,-float('inf')
    trials=0
    for params in candidates:
        va=apply_rhvc_broad(vs,vq,vh,params)
        mrr=evaluate_filtered(va,vq,vf)['MRR']
        trials+=1
        if mrr>best_val:
            best,best_val=params,mrr
    if best is None:raise ValueError('No RHVC candidate configurations')
    adj_valid=apply_rhvc_broad(vs,vq,vh,best)
    adj_test=apply_rhvc_broad(ts,tq,th,best)
    out={'method':'RHVC_broad_history_manuscript_role', 'historical_provenance_verified':False,
         'note':'New broad-history implementation; historical prototype coefficients must be independently matched',
         'backbone':a.backbone,'dataset':a.dataset,'params':asdict(best),
         'configuration_source':selected_by,'configurations_checked':trials,
         'validation_metrics':evaluate_filtered(adj_valid,vq,vf),
         'test_metrics':evaluate_filtered(adj_test,tq,tf)}
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/'rhvc_broad_metrics.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    if a.save_adjusted_scores:
        np.savez_compressed(a.out_dir/'valid_rhvc_scores.npz',scores=adj_valid,triples=vq,entity=np.array(['object']))
        np.savez_compressed(a.out_dir/'test_rhvc_scores.npz',scores=adj_test,triples=tq,entity=np.array(['object']))
    print(json.dumps(out['test_metrics'],indent=2))

if __name__=='__main__':main()
