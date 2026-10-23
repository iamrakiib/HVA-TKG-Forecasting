# Release review - 23 October 2026

Scope: static review of the code-only directory and documentation. Neither the submitted manuscript nor original experimental artifacts were supplied. Findings identify issues to resolve, not evidence of misconduct.

## Issues requiring original records

| Issue | Evidence | Required resolution |
|---|---|---|
| RHVC changed after archived implementation | Broad-history runner and five RHVC wrappers describe replacement of exact-only history | Identify implementation and coefficients that generated reported RHVC values. Label new experiments as new. |
| HRI reference and tied-score ranking | HRI-inspired runner, tie audit, and overlay `metrics.py` | Verify original HRI source, tie policy, configurations, and actual seed runs. Strict-greater ranking assigns rank 1 to all-zero rows. |
| Historical time units unverified | Features subtract stored integer times; timestamp preparer converts to indices | Match original timestamps to the manuscript age definition. |
| Exact reproduction unavailable | Original dumps, checkpoints, logs, diagnostic queries absent | Provide matching artifacts or state narrower release scope. |
| Ownership and citations incomplete | Documentation refers to archives; per-file source references and upstream license inventory absent | Establish ownership, citations, and third-party permissions. MIT applies to author-owned contributions. |
| Environment unrecorded | Unpinned requirements | Recover experiment environment versions. |
| Validation partition semantics | `split_by_time_fraction` splits sorted rows and can divide a snapshot | Compare with submitted validation protocol. |

## Documentation corrected

- Replaced awkward promotional wording with descriptions of functionality and limitations.
- Replaced backbone instructions referencing absent training code and shell scripts with supplied overlay commands.
- Corrected CyGNet documentation to the matrix's object-only evaluation. Its retained branch-average helper is outside that protocol.
- Added the missing plan example, configuration schema, coverage notes, and code map.
- Qualified unsupported claims of complete reproduction and independently verified archive identity.
- Added the requested MIT license and ignored common credential files, local data, and caches.

## Security and validation

The initial 116-file scan found no matches for checked credential patterns (GitHub tokens, common API/AWS keys, private-key headers, and quoted password/secret assignments). No datasets, score dumps, model weights, or result artifacts were found. Pattern scanning cannot guarantee detection of every secret format.

Validation completed using Python 3.10 in an ignored local virtual environment: all 11 existing tests passed. Syntax compilation passed for all 87 Python source files. Local Markdown links, example-plan JSON, and staged Git whitespace checks passed. These checks use synthetic inputs and do not establish benchmark reproduction.

The restricted environment initially reported an expired GitHub login and no Python installation. The normal Windows account provided working GitHub authentication and Python installations; validation used that account. Publication status should be checked on GitHub.
