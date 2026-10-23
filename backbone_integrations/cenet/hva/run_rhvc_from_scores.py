"""RHVC manuscript-aligned broad-history CLI wrapper.

This entry point uses the later broad-history reimplementation; the archived
exact-only implementation is not included in this release.
Historical RHVC MRR cannot be certified without the original weight/config log.
"""
from pathlib import Path
import runpy
import sys
if __name__ == '__main__':
    root=Path(__file__).resolve().parents[3]
    sys.path.insert(0,str(root/'scripts'))
    sys.argv=[sys.argv[0],'--backbone','cenet']+sys.argv[1:]
    runpy.run_path(str(root/'scripts'/'run_rhvc_broad_history.py'),run_name='__main__')
