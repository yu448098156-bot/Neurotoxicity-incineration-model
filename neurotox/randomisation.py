"""Training-only fixed-seed permutation validation with fold-local scaling."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
from joblib import Parallel,delayed,parallel_config
from threadpoolctl import threadpool_limits
from .compare import cv_splits,fit_fold
from .models import build_model
from .metrics import regression_metrics
from .io import write_csv,write_json,prepare_output,sha256
from .explain import checked_training

def one_permutation(index,permutation,X,y,cfg,folds,threads):
    yp=y[permutation]
    name=cfg['selected_model']
    with threadpool_limits(limits=threads):
        full=build_model(name,cfg,threads).fit(X,yp)
        r2=regression_metrics(yp,full.predict(X))['R2']
        cv=[];oof=np.empty_like(y)
        for k,(tr,va) in enumerate(folds,1):
            _,v,p,vm,_=fit_fold(name,cfg,X,yp,tr,va,k,threads)
            cv.append(vm['R2']);oof[v]=p
    return {'run':index,'type':'permuted','train_R2':r2,'CV_R2_mean':float(np.mean(cv)),
            'CV_Q2_pooled':regression_metrics(yp,oof)['R2'],
            **{f'fold_{i+1}_R2':float(x) for i,x in enumerate(cv)}}

def run_randomisation(cfg,bundle,output,iterations=100,seed=42,jobs=1,threads=1,overwrite=False):
    if iterations<1:raise ValueError('iterations must be at least 1')
    prepare_output(output,overwrite)
    started=time.perf_counter()
    train,native,prep,z,md=checked_training(cfg,bundle)
    if cfg['models'][cfg['selected_model']]['params'] != md['configured_parameters']:
        raise ValueError('Randomization model config differs from saved final model')
    folds=cv_splits(train.X,cfg)
    original=one_permutation(0,np.arange(len(train.y)),train.X,train.y,cfg,folds,threads)
    native_r2=regression_metrics(train.y,native.predict(z,thread_count=threads))['R2']
    if abs(original['train_R2']-native_r2)>1e-10:raise RuntimeError('Baseline does not reproduce saved model')
    original['type']='original'
    rng=np.random.default_rng(seed)
    permutations=np.asarray([rng.permutation(len(train.y)) for _ in range(iterations)],dtype=np.int64)
    # Exact zero-based indices permit replay independent of future RNG implementations.
    np.savez_compressed(output/'permutation_indices.npz',indices=permutations)
    write_csv(output/'sample_order.csv',['array_index','record_id'],[{'array_index':i,'record_id':rid} for i,rid in enumerate(train.ids)])
    foldids=np.zeros(len(train.y),int)
    for k,(_,v) in enumerate(folds,1):foldids[v]=k
    write_csv(output/'fold_assignments.csv',['record_id','validation_fold'],[{'record_id':rid,'validation_fold':int(f)} for rid,f in zip(train.ids,foldids)])
    print(f'Y-randomization: {iterations} permutations of training labels, {jobs} workers; held-out test never used.',flush=True)
    with parallel_config(backend='loky',inner_max_num_threads=threads):
        results=Parallel(n_jobs=jobs,verbose=10)(delayed(one_permutation)(i+1,perm,train.X,train.y,cfg,folds,threads) for i,perm in enumerate(permutations))
    write_csv(output/'Y_randomization.csv',list(original),[original]+results)
    p_empirical=(1+sum(r['CV_R2_mean']>=original['CV_R2_mean'] for r in results))/(iterations+1)
    write_json(output/'randomization_manifest.json',{'permutations':iterations,'permutation_seed':seed,
            'numpy_generator':'default_rng / PCG64; exact indices also saved',
            'empirical_p_mean_CV_R2_ge_original':p_empirical,
            'cv':cfg['cv'],'model_sha256':sha256(bundle/'model.cbm'),
            'model_profile':md['configured_parameters'],'elapsed_seconds':time.perf_counter()-started,
            'baseline_reproduces_saved_model':True,'historical_randomized_runs_recovered':False,
            'feature_selection_repeated':False,'scope':'Fixed preselected descriptors/parameters; NOT full workflow including historical feature selection or hyperparameter search.',
            'test_labels_used':False})
