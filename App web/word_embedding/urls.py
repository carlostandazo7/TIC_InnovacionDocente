from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views



urlpatterns = [
    path('signup/', views.SignUp.as_view(), name='signup'),
    path('login/', views.CustomLoginView.as_view(), name='login'),  # Ruta para el login
    path('', views.index, name='index'),
    path('home/', views.home, name='home'),
    path('home/', views.home_view, name='home'),
    path('logout/', auth_views.LogoutView.as_view(next_page='/'), name='logout'),
    path('cargar_documentos/', views.cargar_documentos, name='cargar_documentos'),
    path('ver_documentos/', views.ver_documentos, name='ver_documentos'),

]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)