#!/usr/bin/env bash
set -euo pipefail
DATASET=${DATASET:-ICEWS14}
DATA_ROOT=${DATA_ROOT:-data}
BASE=${BASE:-test_scores_seed42.npz}
HVA=${HVA:-test_adjusted_scores.npz}
python audit_inputs.py --dataset "$DATASET" --data-root "$DATA_ROOT" --dump "$BASE" --split test
python topk_sweep.py --dataset "$DATASET" --data-root "$DATA_ROOT" --base-dump "$BASE" --adjusted-dump "$HVA" --split test
python diagnostic_group_audit.py --dataset "$DATASET" --data-root "$DATA_ROOT" --base-dump "$BASE" --adjusted-dump "$HVA" --split test
python history_age_mechanism.py --dataset "$DATASET" --data-root "$DATA_ROOT" --base-dump "$BASE" --adjusted-dump "$HVA" --split test
python rank_transitions.py --dataset "$DATASET" --data-root "$DATA_ROOT" --base-dump "$BASE" --adjusted-dump "$HVA" --split test
python extract_case_studies.py --dataset "$DATASET" --data-root "$DATA_ROOT" --base-dump "$BASE" --adjusted-dump "$HVA" --split test
