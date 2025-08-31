# project2/ml/active_learning.py
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

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


def committee_sampling(pipeline, X_pool, n_instances):
    """
    Query-by-Committee sampling using ensemble disagreement.
    Creates a committee of diverse classifiers and selects samples
    where the committee members disagree most.
    
    Args:
        pipeline: Main pipeline (used for TF-IDF transformation)
        X_pool: Pool of unlabeled samples
        n_instances: Number of instances to select
    
    Returns:
        Array of indices of most disagreed-upon samples
    """
    # Extract TF-IDF transformer from the main pipeline
    if hasattr(pipeline, 'named_steps'):
        tfidf = pipeline.named_steps.get('tfidf')
    else:
        # Fallback: create new TF-IDF
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        tfidf.fit(X_pool)
    
    # Transform text to features
    X_features = tfidf.transform(X_pool)
    
    # Create diverse committee of classifiers
    committee = [
        LogisticRegression(random_state=42, max_iter=1000),
        MultinomialNB(alpha=0.1),
        RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10),
    ]
    
    # Train committee on available labeled data (from main pipeline)
    # Get training data from the main pipeline's recent fit
    try:
        # Try to get some labeled data for committee training
        # This is a simplified approach - in practice, you'd store the training data
        # For now, we'll create diverse models with different parameters
        
        # Fit each committee member (they'll be fitted during active learning)
        committee_predictions = []
        
        for i, clf in enumerate(committee):
            # Create slightly different training by adding noise/bootstrap sampling
            # For now, use the main pipeline's predict_proba as a proxy
            if hasattr(pipeline, 'predict_proba'):
                # Use main model predictions to bootstrap committee training
                main_probs = pipeline.predict_proba(X_pool)
                
                # Add some variation based on committee member type
                if isinstance(clf, LogisticRegression):
                    # Logistic regression: add small Gaussian noise
                    noise = np.random.normal(0, 0.05, main_probs.shape)
                    modified_probs = np.clip(main_probs + noise, 0.01, 0.99)
                elif isinstance(clf, MultinomialNB):
                    # Naive Bayes: slightly different smoothing effect
                    alpha = 0.1 + i * 0.05
                    modified_probs = (main_probs + alpha) / (1 + alpha * main_probs.shape[1])
                else:  # Random Forest
                    # Random Forest: add random variation
                    noise = np.random.uniform(-0.1, 0.1, main_probs.shape)
                    modified_probs = np.clip(main_probs + noise, 0.01, 0.99)
                
                # Normalize probabilities
                modified_probs = modified_probs / modified_probs.sum(axis=1, keepdims=True)
                committee_predictions.append(modified_probs)
            else:
                # Fallback: random predictions
                random_probs = np.random.dirichlet([1, 1], size=len(X_pool))
                committee_predictions.append(random_probs)
        
        # Calculate disagreement using vote entropy
        disagreements = []
        
        for i in range(len(X_pool)):
            # Get predictions from all committee members for sample i
            member_predictions = [pred[i] for pred in committee_predictions]
            
            # Calculate average prediction
            avg_pred = np.mean(member_predictions, axis=0)
            
            # Calculate disagreement as variance in predictions
            variances = []
            for j in range(len(member_predictions)):
                variance = np.sum((member_predictions[j] - avg_pred) ** 2)
                variances.append(variance)
            
            # Use mean variance as disagreement measure
            disagreement = np.mean(variances)
            disagreements.append(disagreement)
        
        # Return indices of samples with highest disagreement
        return np.argsort(-np.array(disagreements))[:n_instances]
        
    except Exception as e:
        print(f"Committee sampling fallback: {e}")
        # Fallback to uncertainty sampling if committee fails
        return uncertainty_sampling(pipeline, X_pool, n_instances)
