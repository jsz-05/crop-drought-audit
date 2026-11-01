"""Recompute and check the map comparisons and rebuild the paper figures."""
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    for script in ('reproduce_panel.py','verify_kansas.py','spatial_summaries.py',
                   'check_results.py','paper_figures.py'):
        print(f'Running {script}',flush=True)
        subprocess.run([sys.executable,str(ROOT/'analysis'/script)],cwd=ROOT,check=True)
