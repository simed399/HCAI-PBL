import matplotlib
matplotlib.use("Agg")  # headless backend

import base64, io
import pandas as pd
import matplotlib.pyplot as plt

from django.shortcuts import render
from sklearn.model_selection import train_test_split

# classification imports
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import accuracy_score, f1_score, confusion_matrix

# regression imports
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble        import RandomForestRegressor
from sklearn.metrics         import mean_squared_error, r2_score

def index(request):
    context = {
        "column_names":    [],
        "numeric_columns": [],
        "error":           None,
    }

    # 1️⃣ Upload CSV
    if request.method == "POST" and request.FILES.get("csv_file"):
        df = pd.read_csv(request.FILES["csv_file"])
        request.session["csv_data"] = df.to_json()
        context.update({
            "column_names":    df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "table":           df.head().to_html(classes="table", index=False),
        })

    # 2️⃣ Scatter Plot
    elif request.method == "POST" and request.POST.get("action") == "plot":
        df = pd.read_json(request.session["csv_data"])
        context.update({
            "column_names":    df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "table":           df.head().to_html(classes="table", index=False),
        })

        x_col = request.POST["feature_x"]
        y_col = request.POST["feature_y"]
        tgt   = request.POST["target"]

        # factorize for coloring if needed
        if df[tgt].dtype == object:
            df[tgt], _ = pd.factorize(df[tgt])

        fig, ax = plt.subplots()
        sc = ax.scatter(df[x_col], df[y_col], c=df[tgt], cmap="viridis", alpha=0.7)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        plt.colorbar(sc, ax=ax)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        context["plot_url"] = base64.b64encode(buf.read()).decode()

    # 3️⃣ Train Model
    elif request.method == "POST" and request.POST.get("action") == "train":
        df = pd.read_json(request.session["csv_data"])
        context.update({
            "column_names":    df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "table":           df.head().to_html(classes="table", index=False),
        })

        problem   = request.POST["problem_type"]
        model_sel = request.POST["model"]
        test_size = float(request.POST["test_size"])
        hp_val    = float(request.POST["hyperparam"])
        target    = request.POST.get("target_class") or request.POST.get("target_reg")

        # guard: classification only on categorical columns
        if problem == "classification" and df[target].dtype != object:
            context["error"] = (
                f"Cannot classify numeric column “{target}”. "
                "Please pick a categorical target or switch to Regression."
            )
            return render(request, "project1/index.html", context)

        # if regression on categorical, encode the target
        if problem == "regression" and df[target].dtype == object:
            df[target], _ = pd.factorize(df[target])

        # build X from numeric features only
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        X = df[numeric_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # classification branch
        if problem == "classification":
            if model_sel == "LogisticRegression":
                model = LogisticRegression(C=hp_val, max_iter=1000)
            else:
                model = RandomForestClassifier(n_estimators=int(hp_val), random_state=42)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1  = f1_score(y_test, preds, average="weighted")
            cm  = confusion_matrix(y_test, preds)

            # plot confusion matrix
            fig, ax = plt.subplots()
            ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            cm_b64 = base64.b64encode(buf.read()).decode()

            context["train_results"] = {
                "accuracy":         f"{acc:.3f}",
                "f1_score":         f"{f1:.3f}",
                "confusion_matrix": cm_b64,
            }

        # regression branch
        else:
            if model_sel == "LinearRegression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=int(hp_val), random_state=42)

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            r2  = r2_score(y_test, preds)

            # plot actual vs predicted
            fig, ax = plt.subplots()
            ax.scatter(y_test, preds, alpha=0.7)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
            ax.plot(lims, lims, "k--")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            reg_b64 = base64.b64encode(buf.read()).decode()

            context["train_results"] = {
                "mse":      f"{mse:.3f}",
                "r2":       f"{r2:.3f}",
                "reg_plot": reg_b64,
            }

    return render(request, "project1/index.html", context)
