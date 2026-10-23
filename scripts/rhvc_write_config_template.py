#!/usr/bin/env python3
"""Write a blank RHVC specification for user to fill from original run logs.

This template is NOT a historical RHVC experiment setting and is intentionally
non-executable until every coefficient has been supplied.
"""
from pathlib import Path
import json
from run_rhvc_broad_history import RHVCParameters
from dataclasses import fields
import argparse
p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);a=p.parse_args()
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({f.name:None for f in fields(RHVCParameters)},indent=2),encoding='utf8')
print('Populate all fields from your original RHVC configuration before running:',a.out)
