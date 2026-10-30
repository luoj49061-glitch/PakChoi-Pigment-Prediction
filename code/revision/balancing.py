"""The legacy RES ten-bin repetition, applied only inside estimator.fit."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted

def repetition_indices(y):
    y=np.asarray(y,dtype=float).ravel()
    if not len(y) or not np.isfinite(y).all():raise ValueError('Invalid training labels')
    hist,bins=np.histogram(y,bins=10)
    rows=[]
    for i,value in enumerate(y):
        index=max(0,min(np.digitize(value,bins)-1,9))
        repeat=max(1,min(int(hist.max()/(hist[index]+1e-5)),10))
        rows.extend([i]*repeat)
    return np.asarray(rows,dtype=int)

class BinnedRepeatRegressor(RegressorMixin,BaseEstimator):
    def __init__(self,estimator):
        self.estimator=estimator

    def fit(self,X,y):
        rows=repetition_indices(y)
        self.original_n_=len(y);self.expanded_n_=len(rows)
        self.estimator_=clone(self.estimator)
        self.estimator_.fit(np.asarray(X)[rows],np.asarray(y)[rows])
        return self

    def predict(self,X):
        check_is_fitted(self,'estimator_')
        return self.estimator_.predict(X)
