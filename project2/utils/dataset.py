# project2/utils/dataset.py
import os
import pandas as pd
from django.conf import settings
from .preprocessing import clean_text

# Point at your data folder under the project root
DATA_DIR = os.path.join(settings.BASE_DIR, 'imdb_50k')

def load_imdb(split='train', return_original=False):
    """
    Load the IMDB split ('train' or 'test') from data/imdb_50k.
    
    Args:
        split: 'train' or 'test'
        return_original: If True, returns (X_clean, X_original, y)
                        If False, returns (X_clean, y) for backward compatibility
    
    Returns:
        If return_original=False: (X: List[str], y: List[int]) - cleaned text
        If return_original=True: (X_clean: List[str], X_original: List[str], y: List[int])
    """
    fn = os.path.join(DATA_DIR, f"{split}.csv")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Could not find IMDB CSV at {fn}")
    df = pd.read_csv(fn)
    df['clean_review'] = df['review'].apply(clean_text)
    
    X_clean = df['clean_review'].tolist()
    X_original = df['review'].tolist() 
    y = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
    
    if return_original:
        return X_clean, X_original, y
    else:
        return X_clean, y
