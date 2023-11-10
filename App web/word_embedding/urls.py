from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.SignUp.as_view(), name='signup'),
    path('login/', views.CustomLoginView.as_view(), name='login'),  # Ruta para el login
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
]
