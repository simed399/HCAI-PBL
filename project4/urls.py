from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.index, name='index'),
    path('download-guide/', views.download_study_guide, name='download_study_guide'),
    path('task1/', views.task1_intro, name='task1_intro'),
    path('task1/next/', views.task1_next, name='task1_next'),
     # --- study URLs ---
    path('study/start/', views.study_start, name='study_start'),
    path('study/pre/',   views.study_pre_survey,  name='study_pre'),
    path('study/quiz/',  views.study_quiz,        name='study_quiz'),  # Standard study phase (first 10 movies)
    path('study/standard-interest/', views.standard_interest, name='standard_interest'), # Rate standard recommendations
    path('study/guided-start/', views.study_guided_start, name='study_guided_start'), # Guided study transition
    path('study/guided/', views.study_guided,      name='study_guided'), # Guided study phase (second 10 movies)
    path('study/guided-interest/', views.guided_interest, name='guided_interest'), # Rate guided recommendations
    path('study/held/',  views.study_held_out,    name='study_held'),
    path('study/post/',  views.study_post_survey, name='study_post'),
    path('study/thanks/',views.study_thanks,      name='study_thanks'),
    path('study/results/', views.study_results,   name='study_results'),
    path('feedback/',    views.feedback,          name='feedback'),
]