import matplotlib
matplotlib.use("Agg")

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

    # ─── 1️⃣ Upload CSV ───────────────────────────────
    if request.method == "POST" and request.FILES.get("csv_file"):
        df = pd.read_csv(request.FILES["csv_file"])
        request.session["csv_data"] = df.to_json()
        context.update({
            "column_names":    df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "table":           df.head().to_html(classes="table", index=False),
        })

    # ─── 2️⃣ Scatter Plot ─────────────────────────────
    elif request.method == "POST" and request.POST.get("action") == "plot":
        df = pd.read_json(request.session["csv_data"])
        context.update({
            "column_names":    df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "table":           df.head().to_html(classes="table", index=False),
        })
        x = request.POST["feature_x"]
        y = request.POST["feature_y"]
        tgt = request.POST["target"]
        if df[tgt].dtype == object:
            df[tgt], _ = pd.factorize(df[tgt])
        fig, ax = plt.subplots()
        sc = ax.scatter(df[x], df[y], c=df[tgt], cmap="viridis", alpha=0.7)
        ax.set_xlabel(x); ax.set_ylabel(y)
        plt.colorbar(sc, ax=ax)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig); buf.seek(0)
        context["plot_url"] = base64.b64encode(buf.read()).decode()

    # ─── 3️⃣ Train Model ──────────────────────────────
    elif request.method == "POST" and request.POST.get("action") == "train":
        df = pd.read_json(request.session["csv_data"])
        context.update({
            "column_names":    df.columns.tolist(),
            "numeric_columns": df.select_dtypes(include="number").columns.tolist(),
            "table":           df.head().to_html(classes="table", index=False),
        })

        # read common params
        problem   = request.POST["problem_type"]
        model_sel = request.POST["model"]
        test_size = float(request.POST["test_size"])
        context["last_problem"] = problem   # <-- make available to template

        # pick target
        target = request.POST.get("target_class") or request.POST.get("target_reg")

        # guard: classification on categorical only
        if problem == "classification" and df[target].dtype != object:
            context["error"] = f"Cannot classify numeric column “{target}”."
            return render(request, "project1/index.html", context)

        # encode target if needed
        if problem == "regression" and df[target].dtype == object:
            df[target], _ = pd.factorize(df[target])

        # build numeric X
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        X, y = df[num_cols], df[target]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)

        # ─── classification ────────────────────────────
        if problem == "classification":
            if model_sel == "LogisticRegression":
                C        = float(request.POST["lr_C"])
                max_iter = int(request.POST["lr_max_iter"])
                penalty  = request.POST["lr_penalty"]
                solver   = request.POST["lr_solver"]
                model = LogisticRegression(
                    C=C, max_iter=max_iter, penalty=penalty, solver=solver
                )
            else:
                n_est = int(request.POST["rf_n_estimators"])
                max_d = request.POST["rf_max_depth"] or None
                if max_d: max_d = int(max_d)
                min_ss = int(request.POST["rf_min_samples_split"])
                min_sl = int(request.POST["rf_min_samples_leaf"])
                model = RandomForestClassifier(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_ss,
                    min_samples_leaf=min_sl,
                    random_state=42
                )

            model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            acc   = accuracy_score(yte, preds)
            f1    = f1_score(yte, preds, average="weighted")
            cm    = confusion_matrix(yte, preds)

            # plot confusion matrix
            fig, ax = plt.subplots()
            ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i,j], ha="center", va="center")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig); buf.seek(0)
            cm_b64 = base64.b64encode(buf.read()).decode()

            context["train_results"] = {
                "accuracy":         f"{acc:.3f}",
                "f1_score":         f"{f1:.3f}",
                "confusion_matrix": cm_b64,
            }

        # ─── regression ────────────────────────────────
        else:
            if model_sel == "LinearRegression":
                fit_int  = request.POST.get("lin_fit_intercept")=="on"
                copy_x   = request.POST.get("lin_copy_X")=="on"
                positive = request.POST.get("lin_positive")=="on"
                n_jobs   = int(request.POST["lin_n_jobs"])
                model = LinearRegression(
                    fit_intercept=fit_int,
                    copy_X=copy_x,
                    positive=positive,
                    n_jobs=n_jobs
                )
            else:
                n_est = int(request.POST["rf_n_estimators"])
                max_d = request.POST["rf_max_depth"] or None
                if max_d: max_d = int(max_d)
                min_ss = int(request.POST["rf_min_samples_split"])
                min_sl = int(request.POST["rf_min_samples_leaf"])
                model = RandomForestRegressor(
                    n_estimators=n_est,
                    max_depth=max_d,
                    min_samples_split=min_ss,
                    min_samples_leaf=min_sl,
                    random_state=42
                )

            model.fit(Xtr, ytr)
            preds = model.predict(Xte)
            mse   = mean_squared_error(yte, preds)
            r2    = r2_score(yte, preds)

            # plot actual vs predicted
            fig, ax = plt.subplots()
            ax.scatter(yte, preds, alpha=0.7)
            lims = [min(yte.min(), preds.min()), max(yte.max(), preds.max())]
            ax.plot(lims, lims, "k--")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig); buf.seek(0)
            reg_b64 = base64.b64encode(buf.read()).decode()

            context["train_results"] = {
                "mse":      f"{mse:.3f}",
                "r2":       f"{r2:.3f}",
                "reg_plot": reg_b64,
            }

    return render(request, "project1/index.html", context)
