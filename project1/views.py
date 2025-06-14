# project1/views.py

import os
import pandas as pd
import matplotlib.pyplot as plt

from django.conf import settings
from django.core.files.storage import default_storage
from django.shortcuts import render

def index(request):
    """
    Simply render the Project 1 landing page.
    """
    return render(request, 'project1/index.html')


def data_upload(request):
    """
    GET: show the upload form.
    POST: process the uploaded CSV, make a plot, and display it.
    """
    plot_url = None

    if request.method == 'POST' and request.FILES.get('dataset'):
        # 1. Read CSV into a pandas DataFrame
        df = pd.read_csv(request.FILES['dataset'])

        # 2. Drop any 'id' column if present (case-insensitive)
        id_cols = [c for c in df.columns if c.lower() == 'id']
        if id_cols:
            df.drop(columns=id_cols, inplace=True)

        # 3. Separate features vs. target
        feature_cols = df.columns[:-1]
        target_col  = df.columns[-1]

        # 4. Decide whether it's classification or regression
        is_classif = df[target_col].dtype == object or df[target_col].nunique() < 10

        # 5. Build the scatter plot
        plt.figure()
        if is_classif:
            # classification: plot first two features, color by class
            x, y = feature_cols[0], feature_cols[1]
            for cls in df[target_col].unique():
                subset = df[df[target_col] == cls]
                plt.scatter(subset[x], subset[y], label=str(cls))
            plt.xlabel(x)
            plt.ylabel(y)
            plt.legend()
        else:
            # regression: first feature vs. target
            x = feature_cols[0]
            plt.scatter(df[x], df[target_col])
            plt.xlabel(x)
            plt.ylabel(target_col)

        # 6. Save the figure under MEDIA_ROOT/plots/data_plot.png
        fname = 'plots/data_plot.png'
        fpath = os.path.join(settings.MEDIA_ROOT, fname)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        plt.savefig(fpath)
        plt.close()

        # 7. Build a URL for the template to load the image
        plot_url = default_storage.url(fname)

    return render(request, 'project1/data_upload.html', {
        'plot_url': plot_url,
    })
