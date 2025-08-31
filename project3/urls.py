from django.urls import path
from . import views

app_name = "project3"
urlpatterns = [
    path("", views.index, name="index"),
    path("update_tree/", views.update_tree, name="update_tree"),
    path("counterfactuals/", views.generate_counterfactuals, name="counterfactuals"),
]
