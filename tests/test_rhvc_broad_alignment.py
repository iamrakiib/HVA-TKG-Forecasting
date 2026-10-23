"""Independent behavioral checks for the newly written RHVC broader-history runner.

All rows below are synthetic assertions, not empirical results.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'backbone_integrations'/'tirgn'))
from run_rhvc_broad_history import (RHVCParameters, history_signal, apply_rhvc_broad)

PARAMS=dict(gamma=1.,topk=1,stale_threshold=10,lambda_decay=.1,
    exact_occurrence=0.,exact_frequency=0.,exact_recency=0.,exact_stale=0.,
    subject_occurrence=1.,subject_frequency=0.,subject_recency=0.,subject_stale=0.,
    relation_occurrence=2.,relation_frequency=0.,relation_recency=0.,relation_stale=0.)

class TestBroaderRHVC(unittest.TestCase):
    def test_config_cannot_silently_fill_unreported_weights(self):
        with self.assertRaises(ValueError):
            RHVCParameters.from_dict({'gamma':0.1})
        with self.assertRaises(ValueError):
            RHVCParameters.from_dict(dict(PARAMS,exact_recency=float('nan')))
    def test_exact_empty_broader_present(self):
        # The previous archived script would have made NO correction here.
        # Now subject-candidate and relation-candidate evidence both contribute.
        h={'sr':{},'so':{0:{1:[1,2]}},'ro':{0:{1:[1,3]}}}
        scores=np.array([[0.0,4.0,2.0,1.0]],dtype=np.float32)
        query=np.array([[0,0,2,4]],dtype=np.int64)
        p=RHVCParameters.from_dict(PARAMS)
        out=apply_rhvc_broad(scores,query,h,p)
        self.assertAlmostEqual(float(out[0,1]),7.0,places=6)
        np.testing.assert_array_equal(out[0,[0,2,3]],scores[0,[0,2,3]])
    def test_gold_identity_never_enters_candidate_features(self):
        h={'sr':{},'so':{0:{1:[1,2]}},'ro':{0:{1:[1,3]}}}
        s=np.array([[0.,4.,2.,1.]],dtype=np.float32)
        a=np.array([[0,0,0,4]],dtype=np.int64)
        b=np.array([[0,0,3,4]],dtype=np.int64)
        p=RHVCParameters.from_dict(PARAMS)
        np.testing.assert_array_equal(apply_rhvc_broad(s,a,h,p),apply_rhvc_broad(s,b,h,p))
    def test_strict_causality_including_current_time(self):
        no_hist=history_signal([4,4,7],4,10,.1)
        np.testing.assert_array_equal(no_hist,np.zeros(4))
        hist=history_signal([1,1,4,7],4,10,.1)
        self.assertEqual(float(hist[0]),1.)
        self.assertAlmostEqual(float(hist[1]),np.log1p(2))
        self.assertAlmostEqual(float(hist[2]),np.exp(-.1*3))
    def test_history_cannot_create_candidate_outside_k(self):
        # Candidate 2 has stronger history but is outside K=1; it must not change.
        h={'sr':{(0,0):{2:[1]}},'so':{},'ro':{}}
        p=RHVCParameters.from_dict(dict(PARAMS,exact_occurrence=2.))
        s=np.array([[1.,4.,3.,0.]],dtype=np.float32)
        q=np.array([[0,0,2,4]],dtype=np.int64)
        out=apply_rhvc_broad(s,q,h,p)
        np.testing.assert_array_equal(out,s)

if __name__=='__main__':unittest.main()
