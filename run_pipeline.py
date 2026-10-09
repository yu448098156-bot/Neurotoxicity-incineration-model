#!/usr/bin/env python
"""Run the unified study workflow from the repository root; optional full LOO."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('outputs/reproduction'))
    parser.add_argument('--jobs',type=int,default=2)
    parser.add_argument('--loo',action='store_true')
    parser.add_argument('--permutations',type=int,default=100)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    out=(root/args.out).resolve() if not args.out.is_absolute() else args.out
    if out.exists() and any(out.iterdir()):
        parser.error('Use a new, empty output directory. Existing results are never deleted.')
    if args.jobs<1 or args.permutations<1:parser.error('jobs and permutations must be positive')
    env=os.environ.copy()
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        env[name]='1'
    def run(*command):
        subprocess.run([sys.executable,'-m','neurotox',*map(str,command)],cwd=root,env=env,check=True)
    comparison=out/'comparison';explanation=out/'explanation';yrandom=out/'yrandom'
    cmd=['compare','--out',comparison,'--jobs',args.jobs,'--threads',1]
    if args.loo:cmd+=['--loo']
    run(*cmd)
    bundle=comparison/'final_catboost'
    run('explain','--model-dir',bundle,'--out',explanation)
    run('yrandom','--model-dir',bundle,'--out',yrandom,'--iterations',args.permutations,'--jobs',args.jobs)
    run('plots','--comparison',comparison,'--explanation',explanation,'--yrandom',yrandom,'--out',out/'figures')
    run('verify','--model-dir',bundle,'--expected',comparison/'CatBoost_predictions_test.csv','--out',out/'verification.json')
    print(f'Finished. Full outputs: {out}')

if __name__=='__main__':main()
