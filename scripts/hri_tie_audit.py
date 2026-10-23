#!/usr/bin/env python3
"""Explicit rank-tie audit for recurrence-only HRI-inspired score matrices.

The manuscript's rank formula counts candidates strictly ABOVE the gold score.
This gives rank 1 to every gold entity in a completely uninformative all-zero
vector. This file leaves legacy results untouched and computes a SECOND,
clearly labelled expectation under uniformly random ordering of tied candidates.

Expected-uniform metrics average each reciprocal rank / Hits indicator over ALL
possible tie positions; they are not 1/(mean rank). Filtering is applied before
counting gold-score ties. No randomness, sample labels or gold values enter
score construction; labels are used in evaluation only.
"""
from __future__ import annotations
from typing import Dict
import numpy as np

def evaluate_hri_ties(scores, queries, filter_triples, policy='expected_uniform') -> Dict[str,float]:
    from hva.history_utils import build_filter_map_from_triples
    if policy not in ('legacy_strict','expected_uniform'):
        raise ValueError('Policy must be legacy_strict or expected_uniform')
    scores=np.asarray(scores); queries=np.asarray(queries)
    if scores.ndim != 2 or queries.ndim != 2 or len(scores)!=len(queries):
        raise ValueError('scores and queries must have matching rows')
    fmap=build_filter_map_from_triples(filter_triples)
    harmonic=np.cumsum(1.0/np.arange(1,scores.shape[1]+2, dtype=np.float64))
    mrr=hits1=hits3=hits10=0.0
    tied_gold=all_equal=0
    for row,q in zip(scores,queries):
        s,r,o,t=map(int,q[:4]);o=int(o)
        x=np.asarray(row).copy()
        for cand in fmap.get((s,r,t),set()):
            if int(cand)!=o and 0<=int(cand)<len(x):
                x[int(cand)]=-np.inf
        if not np.isfinite(x[o]):
            raise ValueError('Gold score is not finite')
        valid=np.isfinite(x)
        greater=int(np.sum(valid & (x>x[o])))
        equal=int(np.sum(valid & (x==x[o])))
        if equal<1: raise AssertionError('Gold missing from own score tie')
        tied_gold += int(equal>1)
        all_equal += int(np.all(x[valid]==x[o]))
        if policy=='legacy_strict':
            first_rank=greater+1
            mrr+=1.0/first_rank
            hits1+=float(first_rank<=1);hits3+=float(first_rank<=3);hits10+=float(first_rank<=10)
        else:
            # Rank of tied gold is uniformly distributed over g+1 ... g+m.
            g=greater;m=equal
            mrr+=(harmonic[g+m-1]-(harmonic[g-1] if g else 0.0))/m
            for k,v in ((1,'h1'),(3,'h3'),(10,'h10')):
                fraction=float(np.clip(k-g,0,m))/m
                if k==1:hits1+=fraction
                elif k==3:hits3+=fraction
                else:hits10+=fraction
    n=len(queries)
    if n==0: raise ValueError('No queries')
    return {'MRR':mrr/n,'Hits@1':hits1/n,'Hits@3':hits3/n,
            'Hits@10':hits10/n,'count':n,'tie_policy':policy,
            'gold_tied_count':tied_gold,'gold_tied_fraction':tied_gold/n,
            'flat_score_rows':all_equal,'flat_score_fraction':all_equal/n}
