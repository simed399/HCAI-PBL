# project2/utils/dataset.py
import os
import pandas as pd
from django.conf import settings
from .preprocessing import clean_text

# Point at your data folder under the project root
DATA_DIR = os.path.join(settings.BASE_DIR, 'data', 'imdb_50k')

def load_imdb(split='train'):
    """
    Load the IMDB split ('train' or 'test') from data/imdb_50k.
    Returns (X: List[str], y: List[int]).
    """
    fn = os.path.join(DATA_DIR, f"{split}.csv")
    if not os.path.exists(fn):
        raise FileNotFoundError(f"Could not find IMDB CSV at {fn}")
    df = pd.read_csv(fn)
    df['clean_review'] = df['review'].apply(clean_text)
    X = df['clean_review'].tolist()
    y = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
    return X, y
