import numpy as np
import os
import pickle
from .pipeline import make_pipeline
from .active_learning import (
    uncertainty_sampling,
    margin_sampling,
    entropy_sampling,
)
from sklearn.metrics import accuracy_score

def run_experiment(X_pool, y_pool, X_test, y_test, strategy, budget):
    """
    Simulate label acquisition one by one up to `budget`.
    At each step:
      1. Select next index via chosen strategy
      2. "Label" it using y_pool
      3. Retrain pipeline on all selected so far
      4. Record test accuracy
    Returns list of accuracies (len=budget).
    """
    pipeline = make_pipeline()
    labelled_idxs = []
    accuracies = []

    # For an "empty" start, pipeline.predict_proba will error.
    # We can bootstrap with at least one sample from each class:
    unique_classes = np.unique(y_pool)
    if len(unique_classes) < 2:
        raise ValueError("Need at least 2 classes in the dataset")
    
    # Start with one sample from each class
    for class_label in unique_classes:
        class_indices = np.where(np.array(y_pool) == class_label)[0]
        idx = np.random.choice(class_indices)
        labelled_idxs.append(idx)
    
    # Fit pipeline with samples from both classes
    X_initial = [X_pool[i] for i in labelled_idxs]
    y_initial = [y_pool[i] for i in labelled_idxs]
    pipeline.fit(X_initial, y_initial)
    accuracies.append(accuracy_score(y_test, pipeline.predict(X_test)))

    for _ in range(len(unique_classes), budget):
        # build pool of unlabeled
        unlabeled_idxs = [i for i in range(len(X_pool)) if i not in labelled_idxs]
        X_unlabeled = [X_pool[i] for i in unlabeled_idxs]

        # choose next
        if strategy == 'uncertainty':
            pick = uncertainty_sampling
        elif strategy == 'margin':
            pick = margin_sampling
        else:
            pick = entropy_sampling

        next_idx_in_pool = pick(pipeline, X_unlabeled, 1)[0]
        next_idx = unlabeled_idxs[next_idx_in_pool]
        labelled_idxs.append(next_idx)

        # retrain on all so far
        X_lab = [X_pool[i] for i in labelled_idxs]
        y_lab = [y_pool[i] for i in labelled_idxs]
        pipeline.fit(X_lab, y_lab)

        # eval
        acc = accuracy_score(y_test, pipeline.predict(X_test))
        accuracies.append(acc)

    return accuracies
