from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import io
import base64
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np


def plot_tree_image(clf, feature_names):
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=["Adelie", "Chinstrap", "Gentoo"],
        filled=True,
        rounded=True,
        fontsize=10,
        precision=2,
        ax=ax
    )
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def plot_logreg_weights(coef, feature_names):
    mean_abs_weights = np.mean(np.abs(coef), axis=0)
    top_indices = np.argsort(mean_abs_weights)[-10:]
    top_features = [feature_names[i] for i in top_indices]
    top_values = mean_abs_weights[top_indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_features, top_values)
    ax.set_xlabel("Average |Weight| Across Classes")
    ax.set_title("Top Contributing Features (Logistic Regression)")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=200)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def index(request):
    lambda_val = 0.01
    model_type = "tree"

    df = load_penguins().dropna().reset_index(drop=True)
    df_sample = df.head(10).to_html(classes='table', index=False)
    y = df["species"]
    X = pd.get_dummies(df.drop(columns=["species"]))
    y_encoded = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)

    max_leaf_nodes = max(2, int(200 / (lambda_val ** 2)))
    max_leaf_nodes = min(max_leaf_nodes, 100)

    clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    sparsity = clf.get_n_leaves()
    tree_plot = plot_tree_image(clf, X.columns)

    return render(request, 'project3/index.html', {
        'df_sample': df_sample,
        'lambda_val': lambda_val,
        'accuracy': round(float(accuracy) * 100, 2),
        'sparsity': sparsity,
        'tree_plot': tree_plot,
        'model_type': model_type,
        'row_indices': range(10), 
    })


def update_tree(request):
    lambda_val = float(request.GET.get("lambda", 0.01))
    model_type = request.GET.get("model", "tree")

    df = load_penguins().dropna().reset_index(drop=True)
    y = df["species"]
    X = pd.get_dummies(df.drop(columns=["species"]))
    feature_names = list(X.columns)
    y_encoded = LabelEncoder().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.5, random_state=42)

    if model_type == "tree":
        max_leaf_nodes = max(2, int(200 / (lambda_val ** 2)))
        max_leaf_nodes = min(max_leaf_nodes, 100)

        clf = DecisionTreeClassifier(max_leaf_nodes=max_leaf_nodes, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        sparsity = clf.get_n_leaves()
        tree_plot = plot_tree_image(clf, feature_names)

    else:
        C = 1.0 / lambda_val
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=C, max_iter=1000, multi_class="ovr"
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        sparsity = int(np.sum(np.abs(clf.coef_) > 1e-4))
        tree_plot = plot_logreg_weights(clf.coef_, feature_names)

    return JsonResponse({
        'lambda_val': float(lambda_val),
        'accuracy': float(round(accuracy * 100, 2)),
        'sparsity': int(sparsity),
        'tree_plot': tree_plot,
        'model_type': model_type
    })


def generate_counterfactuals(request):
    # Get parameters
    row_index = int(request.GET.get("row", 0))
    target_class = request.GET.get("target", "Adelie")

    # Load and preprocess
    df = load_penguins().dropna().reset_index(drop=True)
    y = df["species"]
    X = df.drop(columns=["species"])
    X_encoded = pd.get_dummies(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Model training
    clf = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=1000)
    clf.fit(X_encoded, y_encoded)

    # Get original x and its prediction
    x_orig = X_encoded.iloc[row_index].values.reshape(1, -1)
    pred_orig = clf.predict(x_orig)[0]
    target_idx = le.transform([target_class])[0]

    # Sample N perturbations around x
    N = 1000
    noise = np.random.normal(0, 0.1, size=(N, X_encoded.shape[1]))
    x_samples = x_orig + noise
    preds = clf.predict(x_samples)

    # Keep only those classified as target
    valid = x_samples[preds == target_idx]
    if len(valid) == 0:
        return JsonResponse({"html_table": "<p>No counterfactuals found for this target class.</p>"})

    # Compute MAD
    mad = np.median(np.abs(X_encoded - np.median(X_encoded, axis=0)), axis=0)
    mad[mad == 0] = 1e-6  # Avoid division by zero

    # Compute MAD-weighted L1 distances
    distances = np.sum(np.abs(valid - x_orig) / mad, axis=1)

    # Select top-k
    k = min(5, len(distances))
    top_indices = np.argsort(distances)[:k]
    top_cf = valid[top_indices]

    # Prepare DataFrame
    df_cf = pd.DataFrame(top_cf, columns=X_encoded.columns)
    rounded_distances = distances[top_indices]
    
    # Ensure it's always treated as an array, even if scalar
    rounded_distances = np.array(rounded_distances, dtype=float).reshape(-1)
    rounded_distances = np.round(rounded_distances, 2)
    
    df_cf.insert(0, "Distance", rounded_distances)

    # Return as HTML
    html_table = df_cf.to_html(index=False, float_format=lambda x: f"{x:.2f}")
    return JsonResponse({"html_table": html_table})
