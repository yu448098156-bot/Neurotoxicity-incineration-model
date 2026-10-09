"""Saved pipelines and a pickle-free CatBoost prediction bundle."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
from .io import environment,sha256,write_json

def save_fitted(pipeline, name: str, cfg: dict, train, test, target: Path) -> dict:
    target.mkdir(parents=True,exist_ok=True)
    artifact=target/'pipeline.joblib'
    joblib.dump(pipeline,artifact,compress=3)
    before=np.asarray(pipeline.predict(test.X),dtype=float)
    restored=joblib.load(artifact)  # Only our own newly written trusted artifact.
    after=np.asarray(restored.predict(test.X),dtype=float)
    err=float(np.max(np.abs(before-after)))
    if not np.allclose(before,after,rtol=1e-12,atol=1e-12):
        raise RuntimeError(f'{name}: predictions changed after save/load')
    scaler=pipeline.named_steps['scaler']
    prep={'schema_version':1,'features':cfg['features'],'mean':scaler.mean_.tolist(),
          'scale':scaler.scale_.tolist(),'var':scaler.var_.tolist(),
          'training_min':np.min(train.X,axis=0).tolist(),'training_max':np.max(train.X,axis=0).tolist(),
          'fitted_on':'training_set.csv only','training_n':len(train.X)}
    write_json(target/'preprocessing.json',prep)
    md={'schema_version':1,'model_name':name,'target':'pLD50','features':cfg['features'],
        'training_n':len(train.X),'test_n_evaluated':len(test.X),'test_used_in_training':False,
        'test_used_for_parameter_selection':False,'selected_as_final':name==cfg['selected_model'],
        'selection_basis':cfg['selection_basis'],'source_script':cfg['models'][name]['source'],
        'configured_parameters':cfg['models'][name]['params'],
        'resolved_estimator_parameters':pipeline.named_steps['model'].get_params(),
        'input_sha256':cfg['expected_input_sha256'],'environment':environment(),
        'reload_max_abs_difference':err,'pipeline_sha256':sha256(artifact),
        'interpretation':'Refactored source-based reference run; not a recovered historical fitted model.',
        'ad_status':'Historical structural applicability-domain algorithm not supplied; no validated AD claim.'}
    if name=='CatBoost':
        pipeline.named_steps['model'].save_model(str(target/'model.cbm'),format='cbm')
        from catboost import CatBoostRegressor
        native=CatBoostRegressor(); native.load_model(str(target/'model.cbm'))
        pred=native.predict((test.X-np.asarray(prep['mean']))/np.asarray(prep['scale']),thread_count=1)
        native_err=float(np.max(np.abs(pred-before)))
        if native_err>1e-12:
            raise RuntimeError('CatBoost native model reload mismatch')
        md.update({'native_reload_max_abs_difference':native_err,'native_model_sha256':sha256(target/'model.cbm'),
                   'resolved_catboost_parameters':pipeline.named_steps['model'].get_all_params()})
    write_json(target/'metadata.json',md)
    files=[p for p in target.iterdir() if p.is_file() and p.name!='bundle_checksums.json']
    write_json(target/'bundle_checksums.json',{p.name:sha256(p) for p in files})
    return {'model':name,'joblib_reload_max_abs_difference':err,
            'native_reload_max_abs_difference':md.get('native_reload_max_abs_difference')}

def load_native(bundle: Path):
    from catboost import CatBoostRegressor
    for name in ['model.cbm','metadata.json','preprocessing.json','bundle_checksums.json']:
        if not (bundle/name).is_file():
            raise FileNotFoundError(f'Missing bundle file: {bundle/name}')
    hashes=json.loads((bundle/'bundle_checksums.json').read_text())
    for name in ['model.cbm','metadata.json','preprocessing.json']:
        if name not in hashes or sha256(bundle/name)!=hashes[name]:
            raise ValueError(f'Bundle checksum mismatch: {name}')
    prep=json.loads((bundle/'preprocessing.json').read_text())
    model=CatBoostRegressor(); model.load_model(str(bundle/'model.cbm'))
    if len(prep['features'])!=len(prep['mean']) or len(prep['features'])!=len(prep['scale']):
        raise ValueError('Invalid preprocessing dimensions')
    if not np.isfinite(prep['mean']).all() or not np.isfinite(prep['scale']).all() or np.any(np.asarray(prep['scale'])<=0):
        raise ValueError('Invalid preprocessing parameters')
    return model,prep

def native_predict(bundle: Path, X: np.ndarray, threads: int = 1) -> np.ndarray:
    model,prep=load_native(bundle)
    if X.ndim!=2 or X.shape[1]!=len(prep['features']) or not np.isfinite(X).all():
        raise ValueError('Incorrect descriptor matrix')
    return np.asarray(model.predict((X-np.asarray(prep['mean']))/np.asarray(prep['scale']),thread_count=threads))
