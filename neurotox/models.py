"""Models preserve uploaded hyperparameters. No automatic algorithm fallback."""
from __future__ import annotations
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

def build_model(name: str, cfg: dict, threads: int = 1):
    if threads < 1:
        raise ValueError('threads must be positive')
    params=dict(cfg['models'][name]['params'])
    if name=='RF':
        estimator=RandomForestRegressor(**params,n_jobs=threads)
    elif name=='GBR':
        estimator=GradientBoostingRegressor(**params)
    elif name=='ETR':
        estimator=ExtraTreesRegressor(**params,n_jobs=threads)
    elif name=='SVR':
        estimator=SVR(**params)
    elif name=='KNN':
        estimator=KNeighborsRegressor(**params,n_jobs=threads)
    elif name=='XGBoost':
        from xgboost import XGBRegressor
        estimator=XGBRegressor(**params,n_jobs=threads)
    elif name=='CatBoost':
        from .catboost_adapter import CatBoostAdapter
        estimator=CatBoostAdapter(parameters=params,thread_count=threads)
    else:
        raise ValueError(f'Unknown model {name}')
    return Pipeline([('scaler',StandardScaler()),('model',estimator)])
