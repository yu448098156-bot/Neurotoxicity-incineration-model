"""Explicit regression metrics; error thresholds are NOT applicability domains."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def regression_metrics(y, pred, p: int | None = None) -> dict:
    y=np.asarray(y,dtype=float); pred=np.asarray(pred,dtype=float)
    if y.shape!=pred.shape or not np.isfinite(y).all() or not np.isfinite(pred).all():
        raise ValueError('Finite observed/predicted vectors of equal shape are required')
    r2=float(r2_score(y,pred))
    result={'R2':r2,'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
            'MAE':float(mean_absolute_error(y,pred))}
    if p is not None:
        result['R2_adjusted_descriptive']=float(1-(1-r2)*(len(y)-1)/(len(y)-p-1)) if len(y)>p+1 else None
    return result

def legacy_error_coverage(train_y,train_pred,test_y,test_pred,multiplier=1.5) -> dict:
    threshold=multiplier*regression_metrics(test_y,test_pred)['RMSE']
    return {'posthoc_threshold_pLD50':float(threshold),
            'train_error_below_threshold_percent':float(100*np.mean(np.abs(train_y-train_pred)<threshold)),
            'test_error_below_threshold_percent':float(100*np.mean(np.abs(test_y-test_pred)<threshold)),
            'interpretation':'Uses observed test errors; retrospective diagnostic only, NOT applicability-domain coverage.'}

def pooled_q2(y,pred) -> float:
    return regression_metrics(y,pred)['R2']
