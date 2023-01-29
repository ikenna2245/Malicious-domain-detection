from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path('url_validator', views.url_validator, name= "url_validator")
] 