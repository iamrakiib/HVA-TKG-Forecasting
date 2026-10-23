# Pack 01: CPU audit + mechanism analyses

No backbone training. Inputs: dataset integer-id files and a base `*.npz` score dump. Optional HVA adjusted score dump.

Score dump schema expected: `scores [N,E]`, `triples [N,4]`, optional `entity`.

Run order:
1. `python audit_inputs.py ...`
2. `python topk_sweep.py ...`
3. `python diagnostic_group_audit.py ...`
4. `python history_age_mechanism.py ...`
5. `python rank_transitions.py ...` (needs adjusted scores)
6. `python extract_case_studies.py ...` (needs adjusted scores)

All history features are causal: VALID uses TRAIN history; TEST uses TRAIN+VALID history. Filtered test evaluation may use all known test truths only for standard filtered ranking labels, never for HVA features.
