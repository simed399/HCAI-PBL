from django.urls import path
from . import views

app_name = 'project2'

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train_full, name='train_full'),
    path('train/start/', views.start_training, name='start_training'),
    path('train/upload/', views.upload_model, name='upload_model'),
    path('active/',     views.active_learning,      name='active_learning'),
    path('active/run/', views.run_active_learning, name='run_active_learning'),
    path('active/results/', views.get_al_results, name='get_al_results'),
    path('experiment/',          views.experiment,        name='experiment'),
    path('experiment/run/',      views.run_experiment,    name='run_experiment'),
    path('experiment/results/',  views.get_exp_results,   name='get_exp_results'),
    path('experiment/live/',     views.show_experiment_live, name='show_experiment_live'),
    path('experiment/data/',     views.get_experiment_data, name='get_experiment_data'),
    path('progress/<str:task>/', views.get_progress, name='get_progress'),
    
    # Human labeling URLs
    path('human-labeling/', views.human_labeling_setup, name='human_labeling_setup'),
    path('human-labeling/start/', views.start_human_labeling, name='start_human_labeling'),
    path('human-labeling/session/<str:session_id>/', views.human_labeling_session, name='human_labeling_session'),
    path('human-labeling/label/<str:session_id>/<int:sample_id>/', views.submit_human_label, name='submit_human_label'),
    path('human-labeling/results/<str:session_id>/', views.human_labeling_results, name='human_labeling_results'),
    path('human-labeling/progress/<str:session_id>/', views.get_session_progress, name='get_session_progress'),
]
