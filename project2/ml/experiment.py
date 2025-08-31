import numpy as np
import os
import pickle
from .pipeline import make_pipeline
from .active_learning import (
    uncertainty_sampling,
    margin_sampling,
    entropy_sampling,
    committee_sampling,
)
from sklearn.metrics import accuracy_score

def run_experiment(X_pool, y_pool, X_test, y_test, strategy, budget, target_accuracy=None, progress_callback=None, 
                  plateau_patience=5, plateau_min_improvement=0.005, enable_plateau_detection=True):
    """
    Simulate label acquisition one by one up to `budget` with stopping conditions.
    At each step:
      1. Select next index via chosen strategy
      2. "Label" it using y_pool
      3. Retrain pipeline on all selected so far
      4. Record test accuracy
      5. Check stopping conditions
      6. Call progress_callback if provided for real-time updates
    
    Stopping conditions:
      - Reach maximum budget
      - Reach target accuracy (if specified)
      - No improvement for several iterations (plateau detection)
    
    Args:
      plateau_patience: Number of iterations without improvement before stopping
      plateau_min_improvement: Minimum improvement threshold (as decimal, e.g., 0.005 for 0.5%)
    
    Returns dict with:
      - accuracies: list of accuracies at each step
      - stop_reason: reason for stopping
      - final_accuracy: last recorded accuracy
      - samples_used: number of samples actually used
    """
    pipeline = make_pipeline()
    labelled_idxs = []
    accuracies = []
    stop_reason = "budget_reached"
    
    # Plateau detection parameters (now configurable)
    patience = plateau_patience  # Number of iterations without improvement
    min_improvement = plateau_min_improvement  # Minimum improvement threshold
    no_improvement_count = 0
    best_accuracy = 0.0

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
    initial_accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    accuracies.append(initial_accuracy)
    best_accuracy = initial_accuracy
    
    # Check if initial accuracy already meets target
    if target_accuracy and initial_accuracy >= target_accuracy:
        stop_reason = "target_accuracy_reached"
        return {
            'accuracies': accuracies,
            'stop_reason': stop_reason,
            'final_accuracy': initial_accuracy,
            'samples_used': len(labelled_idxs)
        }

    for iteration in range(len(unique_classes), budget):
        # build pool of unlabeled
        unlabeled_idxs = [i for i in range(len(X_pool)) if i not in labelled_idxs]
        
        # Check if we've run out of unlabeled samples
        if len(unlabeled_idxs) == 0:
            stop_reason = "no_more_samples"
            break
            
        X_unlabeled = [X_pool[i] for i in unlabeled_idxs]

        # choose next
        if strategy == 'uncertainty':
            pick = uncertainty_sampling
        elif strategy == 'margin':
            pick = margin_sampling
        elif strategy == 'committee':
            pick = committee_sampling
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
        
        # Call progress callback for real-time updates
        if progress_callback:
            progress_callback({
                'iteration': iteration + 1,
                'samples_used': len(labelled_idxs),
                'accuracy': acc,
                'strategy': strategy
            })
        
        # Check stopping conditions
        
        # 1. Target accuracy reached
        if target_accuracy and acc >= target_accuracy:
            stop_reason = "target_accuracy_reached"
            break
        
        # 2. Plateau detection (no significant improvement) - only if enabled
        if enable_plateau_detection:
            if acc > best_accuracy + min_improvement:
                # Significant improvement found
                no_improvement_count = 0
                best_accuracy = acc
            else:
                # No significant improvement
                no_improvement_count += 1
            
            if no_improvement_count >= patience:
                stop_reason = "plateau_detected"
                break
        else:
            # Always update best accuracy when plateau detection is disabled
            if acc > best_accuracy:
                best_accuracy = acc
    
    # If we completed all iterations
    final_accuracy = accuracies[-1] if accuracies else 0.0
    
    return {
        'accuracies': accuracies,
        'stop_reason': stop_reason,
        'final_accuracy': final_accuracy,
        'samples_used': len(labelled_idxs)
    }
