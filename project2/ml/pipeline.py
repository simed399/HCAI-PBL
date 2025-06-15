from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def make_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10_000)),
        ('clf', LogisticRegression(solver='liblinear')),
    ])

def train_and_save(pipeline, X_train, y_train, model_path):
    pipeline.fit(X_train, y_train)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    return pipeline

def load_pipeline(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)
