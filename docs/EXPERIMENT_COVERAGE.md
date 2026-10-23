# Reproducibility and provenance

This release contains five HVA backbone overlays and analysis utilities. It does not include datasets, upstream backbone training source, frozen checkpoints, original score dumps, original run logs, or benchmark outputs. No manuscript was supplied for this release review, so alignment with the submitted version remains unverified.

## Implementation and historical results

- HVA constructs score-only top-K candidates, computes features with strict `t_previous < t_query`, and preserves scores outside top-K. Validation history uses training facts; test history uses training plus validation facts.
- Adapter fitting uses earlier validation rows, with later rows for checkpoint selection. The row-level time-fraction split can divide rows from a single timestamp; confirm whether this matches the submitted protocol.
- Previous documentation described the overlays as copied from supplied archives. Those archives were unavailable in this review, so byte-for-byte identity and ownership were not independently checked.
- Previous release notes state that the three `revision_common.py` helpers were changed to use float32 `torch.topk` to match HVA, including ties. This is a release compatibility change, not evidence about original analyses.
- Launchers, validators, timestamp conversion, runtime profiling, and protocol tests are release utilities. Previous documentation describes synthetic-input validation, not use in producing historical tables.
- Current RHVC uses exact `(s,r,c)`, subject-candidate `(s,*,c)`, and relation-candidate `(*,r,c)` histories. Previous release notes state that archived RHVC was exact-only. The broader-history implementation is a later replacement; original coefficients and run records are needed to identify the implementation that produced reported numbers.
- The HRI-inspired runner is a separate recurrence reference, not official upstream HRI. Deterministic execution cannot establish a five-seed HRI distribution. Strict-greater ranking can assign rank 1 to flat rows; expected-uniform tie sensitivity is reported separately.

## Inputs needed for table reproduction

Retain exact dataset versions and splits, timestamp mappings, backbone checkpoints, validation/test dumps, score normalization, entity branch, seed mapping, hyperparameters, environments, hardware details, and raw run outputs. Diagnostic counts require the actual diagnostic query population; aggregate tables do not identify those queries.

Runtime numbers require measurement on stated hardware. Seed-level significance requires actual matched per-seed outputs. Top-K coverage and inside/outside MRR must share a query population for the weighted-average identity.

Dependencies are unpinned. This release does not claim an independently verified original environment or numerical reproduction of manuscript tables.
