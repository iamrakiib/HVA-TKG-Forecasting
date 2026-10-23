# CyGNet adapter evaluation protocol

The root experiment matrix evaluates original object-prediction queries from frozen CyGNet validation and test score dumps. It does not train CyGNet or average subject and object results.

The retained `hva/combine_branch_results.py` utility computes a separate subject/object branch average. That average is outside this release's object-only matrix protocol and must be labeled separately if used.

Top-K candidates are selected from backbone scores only. Feature lookups use timestamps strictly earlier than the query; validation history is training facts, and test history is training plus validation facts. See the [overlay README](../README_HVA_CYGNET.md) for supplied entry points and the [root documentation](../../../README.md) for input validation.
