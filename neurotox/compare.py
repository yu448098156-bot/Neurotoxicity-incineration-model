"""Common split, descriptors, folds and output schema for all seven regressors."""
from __future__ import annotations
import shutil
import time
from pathlib import Path
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import KFold,LeaveOneOut
from threadpoolctl import threadpool_limits
from .data import load_inputs,data_audit
from .io import write_csv,write_json,environment,prepare_output,sha256
from .metrics import regression_metrics,legacy_error_coverage,pooled_q2
from .models import build_model
from .persistence import save_fitted

def cv_splits(X: np.ndarray,cfg: dict):
    c=cfg['cv']
    return list(KFold(n_splits=c['n_splits'],shuffle=c['shuffle'],
                     random_state=c['random_state'] if c['shuffle'] else None).split(X))

def fit_fold(name,cfg,X,y,tr,va,fold,threads=1):
    with threadpool_limits(limits=threads):
        pipeline=build_model(name,cfg,threads)
        pipeline.fit(X[tr],y[tr])
        pred=np.asarray(pipeline.predict(X[va]),dtype=float)
        train_pred=np.asarray(pipeline.predict(X[tr]),dtype=float)
    return fold,va,pred,regression_metrics(y[va],pred),regression_metrics(y[tr],train_pred)

def fit_loo(name,cfg,X,y,tr,va,threads=1):
    with threadpool_limits(limits=threads):
        pipeline=build_model(name,cfg,threads)
        pipeline.fit(X[tr],y[tr])
        pred=float(pipeline.predict(X[va])[0])
    return int(va[0]),pred

def compare(cfg: dict,output: Path,names:list[str],jobs:int=1,threads:int=1,loo:bool=False,
            overwrite:bool=False,allow_overlap:bool=False) -> None:
    prepare_output(output,overwrite)
    write_json(output/'configuration.json',cfg)
    write_json(output/'environment.json',environment())
    train,test=load_inputs(cfg)
    audit=data_audit(train,test,output/'data_audit',cfg)
    if audit['cross_split_canonical_overlap'] and not allow_overlap:
        raise ValueError('Train/test canonical structures overlap. Audit written; use a reviewed split/config, or explicitly --allow-overlap to reproduce historical data.')
    folds=cv_splits(train.X,cfg)
    fold_id=np.zeros(len(train.X),dtype=int)
    for k,(_,v) in enumerate(folds,1):fold_id[v]=k
    write_csv(output/'cv_assignments.csv',['record_id','source_csv_row','validation_fold'],
              [{'record_id':r['record_id'],'source_csv_row':r['source_csv_row'],'validation_fold':int(k)} for r,k in zip(train.records,fold_id)])
    rows=[]; legacy=[]; reloads=[]
    started=time.perf_counter()
    for name in names:
        tick=time.perf_counter()
        print(f'{name}: training and {len(folds)}-fold CV'+(' + LOO' if loo else ''),flush=True)
        pipeline=build_model(name,cfg,threads)
        with threadpool_limits(limits=threads):
            pipeline.fit(train.X,train.y)
            ptrain=np.asarray(pipeline.predict(train.X))
            ptest=np.asarray(pipeline.predict(test.X))
        trmetrics=regression_metrics(train.y,ptrain,len(cfg['features']))
        temetrics=regression_metrics(test.y,ptest)
        dest=output/'models'/name
        reloads.append(save_fitted(pipeline,name,cfg,train,test,dest))
        oof=np.empty(len(train.y)); cvrows=[]
        with parallel_config(backend='loky',inner_max_num_threads=threads):
            result=Parallel(n_jobs=jobs)(delayed(fit_fold)(name,cfg,train.X,train.y,tr,va,k+1,threads) for k,(tr,va) in enumerate(folds))
        for k,v,p,vm,tm in result:
            oof[v]=p
            cvrows.append({'model':name,'fold':k,'training_n':len(train.y)-len(v),'validation_n':len(v),
                           **{'validation_'+key:val for key,val in vm.items()},
                           **{'training_'+key:val for key,val in tm.items()}})
        write_csv(output/f'{name}_cv_folds.csv',list(cvrows[0]),cvrows)
        write_csv(output/f'{name}_predictions_test.csv',list(test.records[0])+['observed_pLD50','predicted_pLD50','residual_pred_minus_obs','absolute_error'],
                  [{**r,'observed_pLD50':float(y),'predicted_pLD50':float(p),
                    'residual_pred_minus_obs':float(p-y),'absolute_error':float(abs(p-y))} for r,y,p in zip(test.records,test.y,ptest)])
        prows=[{**r,'observed_pLD50':float(y),'in_sample_prediction':float(p),'cv_prediction':float(c),
                'validation_fold':int(f)} for r,y,p,c,f in zip(train.records,train.y,ptrain,oof,fold_id)]
        row={'model':name,'n_train':len(train.y),'n_test':len(test.y),'n_descriptors':len(train.features),
             'R2_train':trmetrics['R2'],'R2_test':temetrics['R2'],'RMSE_train':trmetrics['RMSE'],
             'MAE_train':trmetrics['MAE'],'RMSE_test':temetrics['RMSE'],'MAE_test':temetrics['MAE'],
             'R2_adjusted_train_descriptive':trmetrics['R2_adjusted_descriptive'],
             'Q2_test_equals_R2_test':temetrics['R2'],
             'CV_R2_mean':float(np.mean([r['validation_R2'] for r in cvrows])),
             'CV_R2_SD_sample':float(np.std([r['validation_R2'] for r in cvrows],ddof=1)),
             'CV_RMSE_mean':float(np.mean([r['validation_RMSE'] for r in cvrows])),
             'CV_MAE_mean':float(np.mean([r['validation_MAE'] for r in cvrows])),
             'Q2_CV_pooled':pooled_q2(train.y,oof),'Q2_LOO':None,'LOO_status':'not_run',
             'preselected_final_model':name==cfg['selected_model']}
        if loo:
            with parallel_config(backend='loky',inner_max_num_threads=threads):
                results=Parallel(n_jobs=jobs)(delayed(fit_loo)(name,cfg,train.X,train.y,tr,va,threads) for tr,va in LeaveOneOut().split(train.X))
            ploo=np.empty(len(train.y))
            for i,p in results:ploo[i]=p
            row.update(Q2_LOO=pooled_q2(train.y,ploo),LOO_status='completed')
            for r,p in zip(prows,ploo):r['loo_prediction']=float(p)
        write_csv(output/f'{name}_predictions_training.csv',list(prows[0]),prows)
        row['elapsed_seconds']=float(time.perf_counter()-tick)
        rows.append(row)
        diagnostic=legacy_error_coverage(train.y,ptrain,test.y,ptest,cfg['legacy_error_threshold_multiplier'])
        legacy.append({'model':name,**diagnostic})
        write_csv(output/'model_comparison.csv',list(rows[0]),rows)
        write_csv(output/'legacy_error_diagnostics_NOT_AD.csv',list(legacy[0]),legacy)
        write_csv(output/'save_reload_checks.csv',list(reloads[0]),reloads)
        if name==cfg['selected_model']:
            shutil.copytree(dest,output/'final_catboost',dirs_exist_ok=True)
        print(f"  {name} test R2={row['R2_test']:.6f}; CV mean R2={row['CV_R2_mean']:.6f}; LOO={row['Q2_LOO']}; {row['elapsed_seconds']:.1f}s",flush=True)
    best=max(rows,key=lambda r:r['CV_R2_mean'])['model']
    write_json(output/'run_summary.json',{'completed_models':names,'preselected_final_model':cfg['selected_model'],
               'highest_mean_CV_R2_model':best,'selection_note':'CV best is reported, not used to change the author-prespecified CatBoost model. Test scores are reporting-only.',
               'loo_executed':loo,'elapsed_seconds':time.perf_counter()-started,
               'config_sha256':sha256(output/'configuration.json'),
               'input_audit':audit})
