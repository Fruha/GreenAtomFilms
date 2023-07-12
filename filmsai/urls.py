
from django.urls import path
from filmsai import views

urlpatterns = [
    path('', views.FilmsView.as_view(), name='films_list'),
    path('info/', views.info, name='info'),
    path('film/<int:film_id>/', views.FilmDetailsView.as_view(), name='film_details'),
    path('create/comment/<int:film_id>/', views.AddComment.as_view(), name='add_comment'),
]
