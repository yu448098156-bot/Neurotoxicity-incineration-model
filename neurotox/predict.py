"""Predict new SMILES with the saved CatBoost + stored training scaler."""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
from rdkit import rdBase
from .data import compute_one
from .io import read_csv,write_csv
from .persistence import load_native

def predict(input_path:Path,bundle:Path,output:Path,threads:int=1):
    model,prep=load_native(bundle)
    metadata=json.loads((bundle/'metadata.json').read_text())
    expected=metadata['environment']['packages']['rdkit']
    normalize = lambda v: tuple(int(x) if x.isdigit() else x for x in v.split('.'))
    if normalize(expected)!=normalize(rdBase.rdkitVersion):
        warnings.warn(f'RDKit version differs from saved model: {rdBase.rdkitVersion} versus {expected}. Use the locked environment for reproduction.')
    headers,records=read_csv(input_path)
    if 'SMILES' not in headers:raise ValueError('Prediction CSV must contain SMILES')
    extra=['input_csv_row','predicted_pLD50','prediction_status','error',
           'n_descriptors_outside_training_range','outside_training_range_descriptors']
    if set(headers)&set(extra):raise ValueError('Input contains reserved prediction-output columns')
    rows=[]; X=[]; valid=[]
    for i,r in enumerate(records):
        out={**r,'input_csv_row':i+2,'predicted_pLD50':'','prediction_status':'invalid_input','error':'',
             'n_descriptors_outside_training_range':'','outside_training_range_descriptors':''}
        try:
            x,_=compute_one(r['SMILES'],prep['features'])
            outside=(x<np.asarray(prep['training_min']))|(x>np.asarray(prep['training_max']))
            out['n_descriptors_outside_training_range']=int(outside.sum())
            out['outside_training_range_descriptors']=';'.join(f for f,b in zip(prep['features'],outside) if b)
            X.append(x);valid.append(i)
        except ValueError as exc:out['error']=str(exc)
        rows.append(out)
    if valid:
        z=(np.asarray(X)-np.asarray(prep['mean']))/np.asarray(prep['scale'])
        predictions=model.predict(z,thread_count=threads)
        for i,p in zip(valid,predictions):
            rows[i]['predicted_pLD50']=float(p);rows[i]['prediction_status']='predicted'
    write_csv(output,headers+extra,rows)
    print(f'Predicted {len(valid)}/{len(rows)} rows; invalid rows retained with error messages. Saved {output}')
    print('Training-range flags are descriptive checks, NOT a validated applicability domain or safety decision.')
