from django.shortcuts import render
from django.conf import settings
from .ml.experiment   import run_experiment as run_experiment_sim
import matplotlib.pyplot as plt
from django.http import HttpResponse
from sklearn.metrics import accuracy_score
from .utils.dataset import load_imdb
from .ml.pipeline import make_pipeline, train_and_save, load_pipeline
from .ml.active_learning import (
    uncertainty_sampling, margin_sampling, entropy_sampling
)
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


def active_learning(request):
    """Render form to choose strategy & number of queries."""
    return render(request, 'project2/active_learning.html')

def run_active_learning(request):
    """
    Simulate one pool‐based AL iteration:
    1. Query n samples by chosen strategy
    2. “Label” them via ground-truth
    3. Retrain on just those points
    4. Report test accuracy + show selected samples
    """
    # load pool + test
    X_pool, y_pool = load_imdb('train')
    X_test, y_test = load_imdb('test')

    # parse form
    strategy = request.POST['strategy']
    n_q      = int(request.POST['n_queries'])

    # start from scratch with a small initial dataset
    pipeline = make_pipeline()
    
    # Fit with a small initial dataset (first 10 samples)
    initial_size = min(10, len(X_pool))
    X_initial = X_pool[:initial_size]
    y_initial = y_pool[:initial_size]
    pipeline.fit(X_initial, y_initial)
    
    # Remove initial samples from pool
    X_pool = X_pool[initial_size:]
    y_pool = y_pool[initial_size:]

    # select indices
    if strategy == 'uncertainty':
        idxs = uncertainty_sampling(pipeline, X_pool, n_q)
    elif strategy == 'margin':
        idxs = margin_sampling(pipeline, X_pool, n_q)
    else:
        idxs = entropy_sampling(pipeline, X_pool, n_q)

    # simulate labeling
    X_lab = [X_pool[i] for i in idxs]
    y_lab = [y_pool[i] for i in idxs]

    # retrain & evaluate
    pipeline.fit(X_lab, y_lab)
    acc = accuracy_score(y_test, pipeline.predict(X_test))

    # prepare context
    selected_reviews = [X_pool[i] for i in idxs]
    return render(request, 'project2/active_result.html', {
        'strategy': strategy,
        'n_queries': n_q,
        'accuracy': acc,
        'selected': selected_reviews,
    })

def experiment(request):
    """Form to choose strategy & budget for the full experiment."""
    return render(request, 'project2/experiment.html')

def run_experiment(request):
    """Run the multi–step simulation, plot accuracy vs. #labels, and display."""
    strategy = request.POST['strategy']
    budget   = int(request.POST['budget'])

    # load data
    X_pool, y_pool = load_imdb('train')
    X_test, y_test = load_imdb('test')

    # run sim
    accuracies = run_experiment_sim(X_pool, y_pool, X_test, y_test, strategy, budget)

    # plot to media
    media_subdir = 'project2/experiments'
    media_dir    = os.path.join(settings.MEDIA_ROOT, media_subdir)
    os.makedirs(media_dir, exist_ok=True)
    plot_path    = os.path.join(media_dir, 'accuracy_vs_budget.png')

    plt.figure()
    plt.plot(range(len(accuracies)), accuracies)
    plt.xlabel('Number of labeled samples')
    plt.ylabel('Test accuracy')
    plt.title(f'{strategy.title()} sampling')  # optional
    plt.savefig(plot_path)
    plt.close()

    image_url = settings.MEDIA_URL + f'{media_subdir}/accuracy_vs_budget.png'
    return render(request, 'project2/experiment_result.html', {
        'strategy': strategy,
        'budget': budget,
        'image_url': image_url,
    })