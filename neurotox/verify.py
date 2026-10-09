"""Verification uses exact source IDs and reloads the published model bundle."""
from pathlib import Path
import numpy as np
from .data import load_inputs
from .io import read_csv,write_json,sha256
from .persistence import native_predict

def verify(cfg:dict,bundle:Path,expected:Path,output:Path):
    train,test=load_inputs(cfg)
    _,records=read_csv(expected)
    if [r['record_id'] for r in records]!=test.ids:
        raise ValueError('Recorded test sample IDs/order do not match test_set.csv')
    reference=np.array([float(r['predicted_pLD50']) for r in records])
    observed=np.array([float(r['observed_pLD50']) for r in records])
    if not np.array_equal(observed,test.y):raise ValueError('Recorded observations differ from source pLD50')
    actual=native_predict(bundle,test.X)
    error=float(np.max(np.abs(reference-actual)))
    ok=bool(np.allclose(reference,actual,atol=1e-10,rtol=1e-10))
    report={'passed':ok,'test_n':len(actual),'max_abs_prediction_difference':error,
            'model_sha256':sha256(bundle/'model.cbm'),'test_sha256':cfg['expected_input_sha256']['test_set.csv']}
    write_json(output,report)
    if not ok:raise RuntimeError('Saved-model prediction verification failed')
    return report
