from django.urls import path
from . import views

app_name = "project5"

urlpatterns = [
    path("", views.index, name="index"),

    # training
    path("run-demo", views.run_demo, name="run_demo"),
    path("train-reward", views.train_reward, name="train_reward"),
    path("run-rlhf", views.run_rlhf, name="run_rlhf"),

    # play + status
    path("rollout/<str:kind>", views.rollout_view, name="rollout"),
    path("artifact/<str:name>", views.artifact, name="artifact"),
    path("status", views.status, name="status"),

    # human feedback
    path("pref/new", views.pref_new, name="pref_new"),
    path("pref/choose", views.pref_choose, name="pref_choose"),
]
