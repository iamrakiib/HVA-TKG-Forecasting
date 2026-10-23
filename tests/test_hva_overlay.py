"""Basic deterministic checks; not an end-to-end validation of manuscript metrics."""
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backbone_integrations' / 'tirgn'))
from hva.history_utils import score_only_topk, build_histories
from hva.hva_model import HistoryValidityAdapter, build_hva_features

def main():
    scores = torch.tensor([[0.1, 0.9, 0.2, 0.3]], dtype=torch.float32)
    topk = score_only_topk(scores, 2)
    assert topk.tolist()==[[1,3]], topk.tolist()  # gold id=0 is not inserted
    histories=build_histories([(0,1,3,2),(0,1,3,4),(0,1,2,7)])
    q=torch.tensor([[0,1,-1,5]],dtype=torch.long)
    feat=build_hva_features(q, topk, histories,'cpu','exact_only',10)
    assert feat.shape==(1,2,4)
    assert feat[0,1,0].item()==1.0 # candidate 3 seen before timestamp 5
    assert feat[0,0,0].item()==0.0 # candidate 1 unseen
    model=HistoryValidityAdapter(num_relations=4,mode='exact_only',rel_emb_dim=16,hidden_dim=64,dropout=0.1)
    adjusted,delta=model(torch.gather(scores,1,topk),torch.tensor([1]),feat)
    assert adjusted.shape==(1,2) and delta.shape==(1,2)
    print('PASS: top-K label independence, causal features and HVA input/output shapes')

if __name__=='__main__':main()
