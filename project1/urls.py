from django.urls import path
from . import views

app_name = 'project1'

urlpatterns = [
    path("", views.index, name="index"),
    path('data/', views.data_upload, name='data_upload'),
]