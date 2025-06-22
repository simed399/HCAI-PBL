from django.urls import path
from . import views

app_name = 'project3'

urlpatterns = [
    path('', views.index, name='index'),
    path('update/', views.update_tree, name='update_tree'),
    path('counterfactual/', views.generate_counterfactuals, name='generate_counterfactuals'),

]
