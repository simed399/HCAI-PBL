from django.urls import path
from . import views

app_name = 'project2'

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train_full, name='train_full'),
    path('active/',     views.active_learning,      name='active_learning'),
    path('active/run/', views.run_active_learning, name='run_active_learning'),
    path('experiment/',          views.experiment,        name='experiment'),
    path('experiment/run/',      views.run_experiment,    name='run_experiment'),
]
