"""Native TreeSHAP and fixed-feature ICE from one saved fitted model."""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
from catboost import Pool
from scipy.stats import spearmanr
from .data import load_inputs
from .io import write_csv,write_json,prepare_output,sha256
from .persistence import load_native

def checked_training(cfg,bundle):
    train,test=load_inputs(cfg)
    model,prep=load_native(bundle)
    md=json.loads((bundle/'metadata.json').read_text())
    if cfg['features']!=prep['features']:raise ValueError('Configured and saved descriptor order differ')
    if cfg['expected_input_sha256']!=md['input_sha256']:raise ValueError('Input hashes differ from saved model training inputs')
    z=(train.X-np.asarray(prep['mean']))/np.asarray(prep['scale'])
    return train,model,prep,z,md

def explain(cfg,bundle,output,threads=1,overwrite=False):
    prepare_output(output,overwrite)
    train,model,prep,z,md=checked_training(cfg,bundle)
    values=np.asarray(model.get_feature_importance(Pool(z,label=train.y),type='ShapValues',thread_count=threads))
    base=values[:,-1]; shap_values=values[:,:-1]
    pred=np.asarray(model.predict(z,thread_count=threads))
    maxerr=float(np.max(np.abs(base+shap_values.sum(axis=1)-pred)))
    if maxerr>1e-7:raise RuntimeError('SHAP additivity check failed')
    shaprows=[]
    for i,r in enumerate(train.records):
        for j,f in enumerate(train.features):
            shaprows.append({'record_id':r['record_id'],'source_No':r['source_No'],'CAS':r['CAS'],
                             'descriptor':f,'raw_value':float(train.X[i,j]),'scaled_value':float(z[i,j]),
                             'SHAP_value':float(shap_values[i,j]),'expected_value':float(base[i]),
                             'predicted_pLD50':float(pred[i])})
    write_csv(output/'SHAP_values.csv',list(shaprows[0]),shaprows)
    importance=np.mean(np.abs(shap_values),axis=0)
    order=np.argsort(-importance,kind='stable')
    summary=[]
    for rank,j in enumerate(order,1):
        r,p=(None,None) if np.std(train.X[:,j])==0 or np.std(shap_values[:,j])==0 else spearmanr(train.X[:,j],shap_values[:,j])
        summary.append({'rank':rank,'descriptor':train.features[j],'mean_absolute_SHAP':float(importance[j]),
                        'mean_SHAP':float(np.mean(shap_values[:,j])),
                        'Spearman_raw_descriptor_vs_SHAP':None if r is None else float(r),
                        'Spearman_pvalue_descriptive':None if p is None else float(p)})
    write_csv(output/'SHAP_importance.csv',list(summary[0]),summary)
    write_csv(output/'ICE_curve_ids.csv',list(train.records[0]),train.records)
    pdprows=[];deltarows=[]; counts={}
    for feature in cfg['ice_features']:
        j=train.features.index(feature)
        grid=np.unique(z[:,j])
        varied=np.repeat(z,len(grid),axis=0)
        varied[:,j]=np.tile(grid,len(z))
        curves=np.asarray(model.predict(varied,thread_count=threads)).reshape(len(z),len(grid))
        pdp=curves.mean(axis=0);delta=curves[:,-1]-curves[:,0]
        path=output/f'ICE_{feature}.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.writer(f)
            writer.writerow(['grid_index','descriptor_scaled','descriptor_raw','PDP_mean_pLD50']+train.ids)
            for k,g in enumerate(grid):
                raw=float(g*prep['scale'][j]+prep['mean'][j])
                writer.writerow([k+1,float(g),raw,float(pdp[k])]+curves[:,k].tolist())
                pdprows.append({'descriptor':feature,'grid_index':k+1,'descriptor_scaled':float(g),
                                'descriptor_raw':raw,'PDP_mean_pLD50':float(pdp[k]),'curve_count':len(z)})
        for id,d in zip(train.ids,delta):deltarows.append({'descriptor':feature,'record_id':id,'prediction_end_minus_start':float(d)})
        counts[feature]={'grid_n':len(grid),'curve_n':len(z),'prediction_n':int(curves.size)}
        print(f'ICE {feature}: {len(z)} curves x {len(grid)} grid points',flush=True)
    write_csv(output/'ICE_PDP.csv',list(pdprows[0]),pdprows)
    write_csv(output/'ICE_curve_changes.csv',list(deltarows[0]),deltarows)
    write_json(output/'explanation_manifest.json',{'native_model_sha256':sha256(bundle/'model.cbm'),
               'training_data_sha256':md['input_sha256']['training_set.csv'],
               'model_retrained':False,'SHAP_method':'CatBoost native ShapValues (TreeSHAP)',
               'SHAP_additivity_max_abs_error':maxerr,'SHAP_rows':len(shaprows),
               'ICE_selection':'Four author-specified descriptors, NOT automatic top-four SHAP',
               'ICE_features':cfg['ice_features'],'ICE_counts':counts,
               'historical_figure_identity_claimed':False})
