import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
import os

# Pretrain item embeddings V using SVD as matrix factorization approximation
def train_item_embeddings(k=20):
    # load ratings
    from ..utils.dataset import get_ratings_df
    # pivot to user x item matrix
    ratings_df = get_ratings_df()
    R = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    mat = csr_matrix(R.values)
    svd = TruncatedSVD(n_components=k, random_state=0)
    U = svd.fit_transform(mat)
    V = svd.components_.T  # item embeddings
    # save V
    out = {'V': V, 'movie_ids': R.columns.values}
    np.savez(os.path.join(os.path.dirname(__file__), 'item_embeddings.npz'), **out)
    return V, R.columns.values

# Infer new user vector u from their ratings dict {movieId:rating}
def infer_user_vector(ratings_dict, lamb=0.1):
    import numpy as np
    try:
        # Try to load from saved file first
        data = np.load(os.path.join(os.path.dirname(__file__), 'item_embeddings.npz'))
        V = data['V']           # shape M x K
        ids = data['movie_ids'] # shape M
    except FileNotFoundError:
        # If file doesn't exist, use global embeddings from views
        from ..views import V, movie_ids
        if V is None:
            # If global embeddings not loaded, create dummy ones
            V = np.zeros((10, 20))
            ids = np.arange(10)
        else:
            ids = movie_ids
    
    # build system: minimize ||r - V_u * u||^2 + lamb||u||^2
    # select rows j in ids that are rated
    idxs = []
    rj_values = []
    
    for mid, rating in ratings_dict.items():
        matches = np.where(ids == mid)[0]
        if len(matches) > 0:
            idxs.append(matches[0])
            rj_values.append(rating)
    
    if len(idxs) == 0:
        # If no matches found, return zero vector
        return np.zeros(V.shape[1])
    
    Vj = V[idxs]            # len(S) x K
    rj = np.array(rj_values)
    # solve (Vj^T Vj + lamb*I) u = Vj^T rj
    A = Vj.T.dot(Vj) + lamb * np.eye(V.shape[1])
    b = Vj.T.dot(rj)
    u = np.linalg.solve(A, b)
    return u

def predict_ratings(user_vector, V, movie_ids):
    """Predict ratings for all movies given user vector"""
    predictions = np.dot(V, user_vector)
    return predictions

def get_recommendations(user_vector, V, movie_ids, n=10, exclude_ids=None):
    """Get top N movie recommendations for a user"""
    if exclude_ids is None:
        exclude_ids = []
    
    predictions = predict_ratings(user_vector, V, movie_ids)
    
    # Create array of indices and filter out excluded movies
    valid_indices = []
    for i, movie_id in enumerate(movie_ids):
        if movie_id not in exclude_ids:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return [], []
    
    # Get predictions for valid movies
    valid_predictions = predictions[valid_indices]
    valid_movie_ids = movie_ids[valid_indices]
    
    # Sort by prediction score (descending)
    sorted_indices = np.argsort(valid_predictions)[::-1]
    
    # Return top N
    top_n_indices = sorted_indices[:n]
    recommended_movie_ids = valid_movie_ids[top_n_indices]
    recommended_scores = valid_predictions[top_n_indices]
    
    return recommended_movie_ids, recommended_scores