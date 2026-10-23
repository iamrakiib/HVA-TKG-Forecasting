"""Reproduction-safety checks for seed, timestamp and HRI tie policies."""
import importlib.util
import sys
import tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'backbone_integrations/tirgn'))
from run_experiment_matrix import make_commands,assert_ordinal_timestamps
from hri_tie_audit import evaluate_hri_ties

def test_linear_seed_propagation():
    j={'backbone':'tirgn','dataset':'ICEWS14','seed':2026,
       'data_root':'/data','valid_dump':'/valid.npz','test_dump':'/test.npz'}
    cmd=make_commands(j,Path('/tmp/hva_sample'),False,False,True,False)
    linear=[x for _,x in cmd if 'linear_same_features.py' in x]
    rfs=[x for _,x in cmd if 'rfs_tune_apply.py' in x]
    assert len(linear)==1 and '--seed' in linear[0]
    assert linear[0][linear[0].index('--seed')+1]=='2026'
    assert len(rfs)==1 and '--seed' not in rfs[0]

def test_time_preflight():
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'ICEWS14';p.mkdir()
        (p/'stat.txt').write_text('4 2\n')
        for n,t in [('train',24),('valid',48),('test',72)]:
            (p/f'{n}.txt').write_text(f'0 0 1 {t}\n')
        j={'data_root':tmp,'dataset':'ICEWS14'}
        try: assert_ordinal_timestamps(j)
        except ValueError as e: assert 'not consecutive' in str(e)
        else: raise AssertionError('Raw hour timestamps passed snapshot preflight')
        for n,t in [('train',0),('valid',1),('test',2)]:
            (p/f'{n}.txt').write_text(f'0 0 1 {t}\n')
        assert_ordinal_timestamps(j)

def test_hri_ties():
    q=np.array([[0,0,0,1]])
    filt=q.copy()
    x=np.zeros((1,4),np.float32)
    legacy=evaluate_hri_ties(x,q,filt,policy='legacy_strict')
    fair=evaluate_hri_ties(x,q,filt,policy='expected_uniform')
    assert legacy['MRR']==1 and legacy['Hits@1']==1
    assert abs(fair['MRR']-(1+1/2+1/3+1/4)/4)<1e-10
    assert fair['Hits@1']==.25 and fair['Hits@3']==.75
    assert fair['flat_score_rows']==1 and fair['gold_tied_count']==1
    # Filtered alternative true object must not compete in the tie.
    filt=np.array([[0,0,0,1],[0,0,1,1]])
    fair2=evaluate_hri_ties(x,q,filt,policy='expected_uniform')
    assert fair2['Hits@1']==1/3
    # Unique max agrees with original strict-greater rank.
    x=np.array([[0.,-1.,-2.,-3.]],np.float32)
    assert evaluate_hri_ties(x,q,q,'expected_uniform')['MRR']==1

if __name__=='__main__':
    test_linear_seed_propagation();test_time_preflight();test_hri_ties()
    print('PASS: linear seeds, ordinal preflight, HRI filtered-tie handling')
