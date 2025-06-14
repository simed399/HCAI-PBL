import matplotlib
matplotlib.use("Agg")  # ← headless PNG backend

import base64
import io
import pandas as pd
import matplotlib.pyplot as plt

from django.shortcuts import render
from sklearn.model_selection import train_test_split

# classification imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# regression imports
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def index(request):
    context = {}

    # ─── 1️⃣ Upload & Preview ───────────────────────────────
    if request.method == "POST" and request.FILES.get("csv_file"):
        csv_file = request.FILES["csv_file"]
        try:
            df = pd.read_csv(csv_file)
            # HTML table preview
            context["table"] = df.head().to_html(classes="table", index=False)
            # all column names
            context["column_names"] = df.columns.tolist()
            # only numeric columns for regression
            context["numeric_columns"] = df.select_dtypes(include="number").columns.tolist()
            # stash full data in session
            request.session["csv_data"] = df.to_json()
        except Exception as e:
            context["error"] = str(e)

    # ─── 2️⃣ Scatter Plot ───────────────────────────────────
    elif request.method == "POST" and request.POST.get("action") == "plot":
        df = pd.read_json(request.session["csv_data"])
        x = request.POST["feature_x"]
        y = request.POST["feature_y"]
        target = request.POST["target"]

        fig, ax = plt.subplots()
        sc = ax.scatter(df[x], df[y], c=df[target], cmap="viridis", alpha=0.7)
        ax.set_xlabel(x); ax.set_ylabel(y)
        plt.colorbar(sc, ax=ax)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        context["plot_url"] = base64.b64encode(buf.read()).decode("utf-8")

        # keep preview & dropdowns
        context["table"] = df.head().to_html(classes="table", index=False)
        context["column_names"] = df.columns.tolist()
        context["numeric_columns"] = df.select_dtypes(include="number").columns.tolist()

    # ─── 3️⃣ Train Model ────────────────────────────────────
    elif request.method == "POST" and request.POST.get("action") == "train":
        df = pd.read_json(request.session["csv_data"])

        problem   = request.POST["problem_type"]   # "classification" or "regression"
        model_sel = request.POST["model"]
        test_size = float(request.POST["test_size"])
        hp_val    = float(request.POST["hyperparam"])
        target    = request.POST["target_col"]

        # split features/target
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if problem == "classification":
            # choose classifier
            if model_sel == "LogisticRegression":
                model = LogisticRegression(C=hp_val, max_iter=1000)
            else:
                model = RandomForestClassifier(n_estimators=int(hp_val), random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # metrics
            acc = accuracy_score(y_test, preds)
            f1  = f1_score(y_test, preds, average="weighted")
            cm  = confusion_matrix(y_test, preds)

            # plot confusion matrix
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            cm_b64 = base64.b64encode(buf.read()).decode("utf-8")

            context["train_results"] = {
                "accuracy":         f"{acc:.3f}",
                "f1_score":         f"{f1:.3f}",
                "confusion_matrix": cm_b64,
            }

        else:  # regression
            # choose regressor
            if model_sel == "LinearRegression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=int(hp_val), random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # metrics
            mse = mean_squared_error(y_test, preds)
            r2  = r2_score(y_test, preds)

            context["train_results"] = {
                "mse": f"{mse:.3f}",
                "r2":  f"{r2:.3f}",
            }

        # keep preview & dropdowns after training
        context["table"] = df.head().to_html(classes="table", index=False)
        context["column_names"] = df.columns.tolist()
        context["numeric_columns"] = df.select_dtypes(include="number").columns.tolist()

    return render(request, "project1/index.html", context)
