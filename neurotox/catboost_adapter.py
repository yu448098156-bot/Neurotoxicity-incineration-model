"""Sklearn-compatible wrapper for CatBoost 1.2.8 on sklearn 1.8.

Only API tags/fitted-state interoperability are supplied; training and prediction
are delegated unchanged to the native CatBoostRegressor. No alternate learner
or estimator-level fallback is used.
"""
from __future__ import annotations
import numpy as np
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.utils.validation import check_is_fitted

class CatBoostAdapter(RegressorMixin,BaseEstimator):
    def __init__(self,parameters=None,thread_count=1):
        self.parameters=parameters
        self.thread_count=thread_count

    def fit(self,X,y):
        from catboost import CatBoostRegressor
        self.estimator_=CatBoostRegressor(**(self.parameters or {}),thread_count=self.thread_count)
        self.estimator_.fit(X,y)
        self.n_features_in_=np.asarray(X).shape[1]
        return self

    def predict(self,X):
        check_is_fitted(self,'estimator_')
        return self.estimator_.predict(X,thread_count=self.thread_count)

    def save_model(self,*args,**kwargs):
        check_is_fitted(self,'estimator_')
        return self.estimator_.save_model(*args,**kwargs)

    def get_all_params(self):
        check_is_fitted(self,'estimator_')
        return self.estimator_.get_all_params()
