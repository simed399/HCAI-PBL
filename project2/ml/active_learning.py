# project2/ml/active_learning.py
import numpy as np

def uncertainty_sampling(pipeline, X_pool, n_instances):
    """Select the n_instances where the classifier is least confident"""
    probs = pipeline.predict_proba(X_pool)
    uncertainties = 1 - np.max(probs, axis=1)
    return np.argsort(-uncertainties)[:n_instances]

def margin_sampling(pipeline, X_pool, n_instances):
    """Select the n_instances with smallest difference between top two class probabilities"""
    probs = pipeline.predict_proba(X_pool)
    top2 = np.sort(probs, axis=1)[:, -2:]
    margins = top2[:,1] - top2[:,0]
    return np.argsort(margins)[:n_instances]

def entropy_sampling(pipeline, X_pool, n_instances):
    """Select the n_instances with highest predictive entropy"""
    probs = pipeline.predict_proba(X_pool)
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    return np.argsort(-entropy)[:n_instances]
