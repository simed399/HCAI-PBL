import io
import base64
import numpy as np
import pandas as pd
from dataclasses import dataclass

from django.http import JsonResponse, HttpResponseBadRequest
from django.shortcuts import render

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression

# ---- Data loading -----------------------------------------------------------

def _load_penguins() -> pd.DataFrame:
    # requires: pip install palmerpenguins
    from palmerpenguins import load_penguins
    df = load_penguins()
    # keep only columns we use; drop rows with missing values
    cols = [
        "species", "island", "sex",
        "bill_length_mm", "bill_depth_mm",
        "flipper_length_mm", "body_mass_g"
    ]
    df = df[cols].dropna().reset_index(drop=True)
    return df

DF = _load_penguins()
TARGET = "species"
CAT_COLS = ["island", "sex"]
NUM_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

# Precompute a nice sample for the UI
DF_SAMPLE_HTML = (
    DF.head(10)
      .to_html(index=False, classes="table table-sm table-striped table-hover")
)

ROW_INDICES_FOR_UI = list(DF.index[:10])  # shown in the dropdown

# ---- Helpers ----------------------------------------------------------------

def _img_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _lambda_to_tree_leaves(lambda_val: float) -> int:
    # map λ ∈ [1e-4, 1.0] to leaves ∈ [30, 2] (higher λ → sparser)
    lam = max(1e-4, min(float(lambda_val), 1.0))
    leaves = int(np.round(np.interp(lam, [1e-4, 1.0], [30, 2])))
    return max(2, leaves)

def _build_preprocessor(for_model: str) -> ColumnTransformer:
    if for_model == "logreg":
        num_pipe = Pipeline([("scaler", StandardScaler())])
    else:
        num_pipe = "passthrough"
    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([
        ("num", num_pipe, NUM_COLS),
        ("cat", cat_pipe, CAT_COLS),
    ])

@dataclass
class Trained:
    pipe: Pipeline
    features: list[str]
    accuracy_pct: float
    sparsity: int
    model_type: str  # "tree" or "logreg"

def _feature_names_from_preprocessor(pre: ColumnTransformer) -> list[str]:
    names = []
    # numeric
    names.extend(NUM_COLS)
    # categoricals from OHE
    cat_enc = pre.named_transformers_["cat"]
    cat_names = list(cat_enc.get_feature_names_out(CAT_COLS))
    names.extend(cat_names)
    return names

def _train_model(lambda_val: float = 0.01, model_type: str = "tree") -> Trained:
    model_type = "logreg" if model_type == "logreg" else "tree"

    X = DF[CAT_COLS + NUM_COLS]
    y = DF[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pre = _build_preprocessor(model_type)

    if model_type == "tree":
        leaves = _lambda_to_tree_leaves(lambda_val)
        model = DecisionTreeClassifier(
            random_state=42, max_leaf_nodes=leaves, class_weight=None
        )
    else:
        lam = max(1e-4, min(float(lambda_val), 1.0))
        C = max(1e-3, min(1.0 / lam, 1e4))
        model = LogisticRegression(
            penalty="l1", C=C, solver="liblinear", multi_class="ovr", max_iter=2000
        )

    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test) * 100.0

    # sparsity
    m = pipe.named_steps["model"]
    if model_type == "tree":
        sparsity = m.get_n_leaves()
    else:
        # count non-zero weights across all classes
        sparsity = int(np.count_nonzero(m.coef_))

    # derive feature names (after fit)
    features = _feature_names_from_preprocessor(pipe.named_steps["pre"])
    return Trained(pipe, features, round(acc, 2), sparsity, model_type)

def _plot_tree_image(tr: Trained) -> str:
    assert tr.model_type == "tree"
    tree = tr.pipe.named_steps["model"]
    class_names = list(tree.classes_)
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        tree, ax=ax, filled=True, feature_names=tr.features,
        class_names=class_names, proportion=True, fontsize=8
    )
    return _img_to_base64(fig)

def _plot_logreg_weights_image(tr: Trained, top_k: int = 12) -> str:
    assert tr.model_type == "logreg"
    lr = tr.pipe.named_steps["model"]
    # max weight magnitude across classes
    w = np.max(np.abs(lr.coef_), axis=0)
    idx = np.argsort(w)[-top_k:][::-1]
    labels = [tr.features[i] for i in idx]
    vals = w[idx]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels[::-1], vals[::-1])
    ax.set_xlabel("Absolute weight (max across classes)")
    ax.set_title("Top model features")
    fig.tight_layout()
    return _img_to_base64(fig)

# ---- Counterfactuals --------------------------------------------------------

def _mad(series: pd.Series) -> float:
    med = np.median(series)
    return float(np.median(np.abs(series - med))) or 1.0

def _sample_counterfactuals(row_idx: int, target_label: str, n: int = 400,
                            base_lambda: float = 0.01, use_model: str = "logreg") -> pd.DataFrame:
    """Generate n local perturbations around a row; return a table of the
    best counterfactuals according to MAD-weighted L1 distance."""
    tr = _train_model(base_lambda, use_model)
    x0 = DF.loc[row_idx, CAT_COLS + NUM_COLS].copy()

    # numeric perturbations (Laplace around original, scaled by MAD of each column)
    mads = {c: _mad(DF[c].values) for c in NUM_COLS}
    num_samples = {
        c: np.clip(
            x0[c] + np.random.laplace(0.0, mads[c] * 0.2, size=n),
            DF[c].min(), DF[c].max()
        )
        for c in NUM_COLS
    }

    # categorical perturbations: keep original w.p. 0.6 else sample from empirical distribution
    rng = np.random.default_rng(0)
    cat_samples = {}
    for c in CAT_COLS:
        choices, probs = np.unique(DF[c].values, return_counts=True)
        probs = probs / probs.sum()
        draws = rng.choice(choices, size=n, p=probs)
        mask_keep = rng.random(n) < 0.6
        draws[mask_keep] = x0[c]
        cat_samples[c] = draws

    X_new = pd.DataFrame({**cat_samples, **num_samples})
    preds = tr.pipe.predict(X_new)
    keep = preds == target_label
    if not np.any(keep):
        return pd.DataFrame(columns=["distance", "proba_target", *X_new.columns])

    # distance: MAD-weighted L1 on numeric + 1 for each changed categorical
    dist_num = np.zeros(np.sum(keep))
    Xk = X_new.loc[keep].reset_index(drop=True)
    for c in NUM_COLS:
        dist_num += np.abs(Xk[c] - x0[c]) / mads[c]
    dist_cat = sum((Xk[c] != x0[c]).astype(int) for c in CAT_COLS)
    distance = dist_num + dist_cat

    # score: probability of target
    if hasattr(tr.pipe.named_steps["model"], "predict_proba"):
        proba = tr.pipe.predict_proba(Xk)  # shape (n, n_classes)
        classes = list(tr.pipe.named_steps["model"].classes_)
        t_idx = classes.index(target_label)
        proba_t = proba[:, t_idx]
    else:
        proba_t = np.ones(len(Xk))  # trees without predict_proba fallback

    out = Xk.copy()
    out.insert(0, "proba_target", np.round(proba_t, 3))
    out.insert(0, "distance", np.round(distance, 3))

    # keep top 8 closest
    out = out.sort_values(["distance", "proba_target"], ascending=[True, False]).head(8)
    return out

def _df_to_bootstrap_table(df: pd.DataFrame) -> str:
    if df.empty:
        return '<div class="text-muted">No counterfactuals found nearby. Try a different row/target.</div>'
    return df.to_html(index=False, classes="table table-sm table-striped table-hover")

# ---- Views ------------------------------------------------------------------

def index(request):
    lambda_val = float(request.GET.get("lambda", 0.01))
    model_type = request.GET.get("model", "tree")

    tr = _train_model(lambda_val, model_type)
    if tr.model_type == "tree":
        img = _plot_tree_image(tr)
    else:
        img = _plot_logreg_weights_image(tr)

    return render(request, "project3/index.html", {
        "lambda_val": lambda_val,
        "model_type": tr.model_type,
        "accuracy": tr.accuracy_pct,
        "sparsity": tr.sparsity,
        "tree_plot": img,
        "df_sample": DF_SAMPLE_HTML,
        "row_indices": ROW_INDICES_FOR_UI,
    })

from django.http import JsonResponse

def _to_py_number(x):
    # works for numpy scalars and pandas dtypes
    try:
        return x.item()
    except AttributeError:
        return x

def update_tree(request):
    lambda_val = float(request.GET.get("lambda", "0.01"))
    model_type = request.GET.get("model", "tree")

    tr = _train_model(lambda_val, model_type)

    if tr.model_type == "tree":
        img = _plot_tree_image(tr)
    else:
        img = _plot_logreg_weights_image(tr)

    # Cast everything to built-in Python types
    data = {
        "accuracy": round(float(_to_py_number(tr.accuracy_pct)), 2),
        "sparsity": int(_to_py_number(tr.sparsity)),
        "model_type": str(tr.model_type),
        "tree_plot": img,  # already a base64 string
    }
    return JsonResponse(data)


def generate_counterfactuals(request):
    try:
        row = int(request.GET.get("row", 0))
        target = request.GET.get("target", DF[TARGET].unique().tolist()[0])
    except ValueError:
        return HttpResponseBadRequest("Invalid parameters")

    if row < 0 or row >= len(DF):
        return HttpResponseBadRequest("Row out of range")

    table = _sample_counterfactuals(row, target, n=400, base_lambda=0.01, use_model="logreg")
    html = _df_to_bootstrap_table(table)
    return JsonResponse({"html_table": html})
