"""Deterministic unit tests of paper protocol; no benchmark data or outcomes."""
from __future__ import annotations
import tempfile,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_ordinal_snapshots import normalize_splits,convert_dump
sys.path.insert(0,str(ROOT/'backbone_integrations/tirgn'))
from hva.history_utils import build_histories,score_only_topk
from hva.hva_model import HistoryValidityAdapter,apply_hva_batch,build_hva_features

def test_protocol():
    # The target must not be introduced to the score-only top-K candidate set.
    scores=torch.tensor([[0.20,0.91,0.31,0.01]],dtype=torch.float32)
    assert score_only_topk(scores,2).tolist()==[[1,2]]
    gold_id=3
    assert gold_id not in score_only_topk(scores,2)[0].tolist()
    # Strict causality with duplicate event records at the same timestamp.
    histories=build_histories([(0,1,1,1),(0,1,1,1),(0,1,1,3),(0,1,2,9)])
    q=np.array([[0,1,3,4]],dtype=np.int64)
    ids=score_only_topk(scores,2)
    feat=build_hva_features(q,ids,histories,'cpu','exact_only',2)
    assert feat.shape==(1,2,4)
    assert np.isclose(feat[0,0,2].item(),np.log1p(3))  # records, not unique timestamps
    assert feat[0,0,3].item()==0.0  # last snapshot at 3; gap 1
    assert torch.all(feat[0,1]==0)  # candidate 2 observed only in the future
    model=HistoryValidityAdapter(num_relations=4,mode='exact_only',rel_emb_dim=16,
                                 hidden_dim=64,dropout=.1,gamma=.1)
    with torch.no_grad():
        final,delta_top,top_ids,delta=apply_hva_batch(model,scores,q,histories,2,'cpu')
    assert final.shape==(1,4)
    assert final[0,0]==scores[0,0] and final[0,3]==scores[0,3]
    assert model.mlp[0].in_features==23
    dual=HistoryValidityAdapter(num_relations=4,mode='dual_branch')
    assert dual.mlp[0].in_features==31
    assert -1.0<torch.tanh(model.gamma_raw).item()<1.0

def test_time_indexing():
    with tempfile.TemporaryDirectory() as temp:
        base=Path(temp);data=base/'original'/'ICEWS14';dest=base/'normalized'/'ICEWS14';data.mkdir(parents=True);dest.mkdir(parents=True)
        arrays={'train':[(0,0,1,100),(0,0,1,124)],
                'valid':[(0,0,2,148)],'test':[(0,0,1,220)]}
        for name,rows in arrays.items():
            np.savetxt(data/f'{name}.txt',np.asarray(rows,dtype=int),fmt='%d')
        (data/'stat.txt').write_text('4 1\n',encoding='utf8')
        mapping=normalize_splits(data,dest)
        assert mapping=={100:0,124:1,148:2,220:3}
        test= np.array(arrays['test'],dtype=int)
        np.savez(base/'test.npz',scores=np.zeros((1,4),np.float32),triples=test,entity=np.array(['object']))
        convert_dump(base/'test.npz',base/'converted.npz',mapping)
        with np.load(base/'converted.npz') as loaded:assert loaded['triples'][0,3]==3

def test_analysis_topk_matches_training():
    sys.path.insert(0,str(ROOT/'analysis/revision_tools/01_CPU_AUDIT_AND_MECHANISM'))
    from revision_common import topk_indices
    x=np.array([[0.1,0.4,0.7,0.0],[0.9,0.3,0.2,0.1]],np.float32)
    assert np.array_equal(topk_indices(x,2),score_only_topk(torch.from_numpy(x),2).numpy())

if __name__=='__main__':
    test_protocol();test_time_indexing();test_analysis_topk_matches_training()
    print('PASS: causality, count, top-K, input dimensions, gate, no-gold insertion, ordinal conversion')
