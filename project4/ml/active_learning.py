import numpy as np
from .mf import infer_user_vector

def uncertainty_query(current_u, V, movie_ids, rated_ids, scale_mid=2.5, n=1):
    # predict score = u^T v for each unrated item; pick closest to midpoint
    unrated_mask = ~np.isin(movie_ids, rated_ids)
    V_un = V[unrated_mask]
    ids_un = movie_ids[unrated_mask]
    preds = V_un.dot(current_u)
    uncertainty = np.abs(preds - scale_mid)
    idxs = np.argsort(uncertainty)[:n]
    return ids_un[idxs]