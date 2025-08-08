from django.urls import path
from . import views

app_name = 'project4'

urlpatterns = [
    path('', views.index, name='index'),
    path('task1/', views.task1_intro, name='task1_intro'),
    path('task1/next/', views.task1_next, name='task1_next'),
     # --- study URLs ---
    path('study/start/', views.study_start, name='study_start'),
    path('study/pre/',   views.study_pre_survey,  name='study_pre'),
    path('study/quiz/',  views.study_quiz,        name='study_quiz'),  # wrappers True->Task1, False->control
    path('study/held/',  views.study_held_out,    name='study_held'),
    path('study/post/',  views.study_post_survey, name='study_post'),
    path('study/thanks/',views.study_thanks,      name='study_thanks'),
]