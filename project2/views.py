from django.shortcuts import render
from django.http import HttpResponse
from sklearn.metrics import accuracy_score
from .utils.dataset import load_imdb
from .ml.pipeline import make_pipeline, train_and_save, load_pipeline
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model_full.pkl')

def index(request):
    return render(request, 'project2/index.html')

def train_full(request):
    # Load data
    X_train, y_train = load_imdb('train')
    X_test,  y_test  = load_imdb('test')

    # Train or load
    if os.path.exists(MODEL_PATH):
        pipeline = load_pipeline(MODEL_PATH)
    else:
        pipeline = make_pipeline()
        pipeline = train_and_save(pipeline, X_train, y_train, MODEL_PATH)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return render(request, 'project2/train.html', {'accuracy': acc})
