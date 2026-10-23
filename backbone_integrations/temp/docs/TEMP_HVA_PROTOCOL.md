# TeMP adapter evaluation protocol

Train TeMP and export frozen object-prediction validation and test scores using an external upstream implementation. This release supplies adapters, not the TeMP encoder, training scripts, or a patched score exporter.

Validate dumps and timestamp units using the root utilities, then apply `python -m hva.run_hva_from_scores` from this overlay directory. See the [overlay README](../README_HVA_TeMP.md) for a full command.

The adapter selects candidates using backbone scores only and computes historical features with strict `t_previous < t_query`. Validation uses training history; test uses training plus validation history. HVA exact-only and dual-branch are ranking-stage corrections over frozen scores.
