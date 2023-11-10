from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from .forms import DocumentForm

# Vista para el registro de usuarios
class SignUp(generic.CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login')
    template_name = 'registration/signup.html'

# Vista para la página de inicio
def index(request):
    return render(request, 'index.html')

# Vista para el login
class CustomLoginView(LoginView):
    template_name = 'registration/login.html' 
    
    
@login_required
def home(request):
    return render(request, 'home.html')

