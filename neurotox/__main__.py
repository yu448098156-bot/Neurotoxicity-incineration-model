"""Command-line entry points; importing the package never trains a model."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from .io import ROOT,load_config

def parser():
    p=argparse.ArgumentParser(description='Reproducible, source-based neurotoxicity model comparison.')
    p.add_argument('--config',type=Path,default=ROOT/'config/models.json')
    s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('audit',help='Validate fixed inputs and descriptor/structure overlap without training.')
    a.add_argument('--out',type=Path,default=ROOT/'outputs/audit')
    a=s.add_parser('compare',help='Train, compare and save seven source-configured regressors.')
    a.add_argument('--out',type=Path,default=ROOT/'outputs/comparison')
    a.add_argument('--models',nargs='+',default=None)
    a.add_argument('--jobs',type=int,default=1,help='Parallel CV/LOO workers. Use 1 for a low-memory computer.')
    a.add_argument('--threads',type=int,default=1)
    a.add_argument('--loo',action='store_true',help='Run actual LOO fits for all requested models (slow).')
    a.add_argument('--overwrite',action='store_true')
    a.add_argument('--allow-overlap',action='store_true')
    a=s.add_parser('predict',help='Predict from a saved native CatBoost model; never retrains.')
    a.add_argument('--input',type=Path,required=True)
    a.add_argument('--model-dir',type=Path,default=ROOT/'models/final_catboost')
    a.add_argument('--out',type=Path,default=ROOT/'outputs/predictions.csv')
    a.add_argument('--threads',type=int,default=1)
    a=s.add_parser('explain',help='SHAP and four fixed ICE descriptors from one saved CatBoost.')
    a.add_argument('--model-dir',type=Path,default=ROOT/'models/final_catboost')
    a.add_argument('--out',type=Path,default=ROOT/'outputs/explanation')
    a.add_argument('--threads',type=int,default=1)
    a.add_argument('--overwrite',action='store_true')
    a=s.add_parser('yrandom',help='Training-only Y-permutation test with saved seed and fold indices.')
    a.add_argument('--model-dir',type=Path,default=ROOT/'models/final_catboost')
    a.add_argument('--out',type=Path,default=ROOT/'outputs/yrandom')
    a.add_argument('--iterations',type=int,default=100)
    a.add_argument('--seed',type=int,default=42)
    a.add_argument('--jobs',type=int,default=1)
    a.add_argument('--threads',type=int,default=1)
    a.add_argument('--overwrite',action='store_true')
    a=s.add_parser('plots',help='Draw each panel separately from exported numeric CSVs.')
    a.add_argument('--comparison',type=Path,default=ROOT/'outputs/comparison')
    a.add_argument('--explanation',type=Path,default=None)
    a.add_argument('--yrandom',type=Path,default=None)
    a.add_argument('--out',type=Path,default=ROOT/'outputs/figures')
    a=s.add_parser('verify',help='Check final-bundle integrity and reproduce recorded test predictions.')
    a.add_argument('--model-dir',type=Path,default=ROOT/'models/final_catboost')
    a.add_argument('--expected',type=Path,default=ROOT/'reference_results/CatBoost_predictions_test.csv')
    a.add_argument('--out',type=Path,default=ROOT/'outputs/verification.json')
    return p

def main():
    args=parser().parse_args()
    cfg=load_config(args.config)
    if getattr(args,'jobs',1)<1 or getattr(args,'threads',1)<1:
        raise ValueError('jobs and threads must be positive integers')
    if args.command=='audit':
        from .data import load_inputs,data_audit
        tr,te=load_inputs(cfg)
        print(data_audit(tr,te,args.out,cfg))
    elif args.command=='compare':
        from .compare import compare
        names=args.models or cfg['model_order']
        if len(set(names))!=len(names) or set(names)-set(cfg['models']):raise ValueError('Invalid model list')
        compare(cfg,args.out,names,args.jobs,args.threads,args.loo,args.overwrite,args.allow_overlap)
    elif args.command=='predict':
        from .predict import predict
        predict(args.input,args.model_dir,args.out,args.threads)
    elif args.command=='explain':
        from .explain import explain
        explain(cfg,args.model_dir,args.out,args.threads,args.overwrite)
    elif args.command=='yrandom':
        from .randomisation import run_randomisation
        run_randomisation(cfg,args.model_dir,args.out,args.iterations,args.seed,args.jobs,args.threads,args.overwrite)
    elif args.command=='plots':
        from .plots import plot_all
        plot_all(args.comparison,args.explanation,args.yrandom,args.out)
    elif args.command=='verify':
        from .verify import verify
        print(verify(cfg,args.model_dir,args.expected,args.out))

if __name__=='__main__':
    try:main()
    except (ValueError,FileNotFoundError,FileExistsError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(2)
