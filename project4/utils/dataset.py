import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'movielens')

# Global variables for lazy loading
_ratings_df = None
_movies_df = None

def _load_data():
    """Load data files if not already loaded"""
    global _ratings_df, _movies_df
    if _ratings_df is None or _movies_df is None:
        try:
            _ratings_df = pd.read_csv(os.path.join(DATA_DIR, 'ratings.csv'))
            _movies_df = pd.read_csv(os.path.join(DATA_DIR, 'movies.csv'))
        except FileNotFoundError:
            # Create dummy data if files don't exist
            _ratings_df = pd.DataFrame({'userId': [], 'movieId': [], 'rating': [], 'timestamp': []})
            _movies_df = pd.DataFrame({'movieId': [], 'title': [], 'genres': []})

def get_movie_list():
    """Return list of (movieId, title)"""
    _load_data()
    return list(_movies_df[['movieId', 'title']].itertuples(index=False, name=None))

def get_ratings_df():
    """Get ratings dataframe"""
    _load_data()
    return _ratings_df

def get_movies_df():
    """Get movies dataframe"""
    _load_data()
    return _movies_df