from .models import Documento  # Asegúrate de importar tu modelo Documento
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from django.shortcuts import render, redirect
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from .forms import *
import pandas as pd
import spacy
import io
import re

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

@login_required
def home_view(request):
    return render(request, 'home.html', {'user': request.user})

# Vista para el preprocesamiento de los datos

def eliminar_stop_words(texto):
    nlp = spacy.load('es_core_news_sm')
    doc = nlp(texto)
    palabras_filtradas = [token.text for token in doc if not token.is_stop]
    return ' '.join(palabras_filtradas)

def cargar_documentos(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            documento = form.save(commit=False)

            # Obtén el archivo del formulario
            archivo = request.FILES['archivo']

            # Lee el contenido del archivo XLSX en un DataFrame de pandas
            df = pd.read_excel(archivo)

            # Realizar las operaciones de limpieza
            df = df.dropna()
            df = df.drop_duplicates()
            df['texto'] = df['texto'].str.lower()

            palabras_a_eliminar = ['&nbsp;', 'pueden', 'traves', 'vez', 'tal', 'así', 'ademas']
            for palabra in palabras_a_eliminar:
                df['texto'] = df['texto'].str.replace(palabra, '')

            df['texto'] = df['texto'].str.lstrip()
            df['texto'] = df['texto'].str.replace('\d+', '', regex=True)
            df['texto'] = df['texto'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
            df['texto'] = df['texto'].apply(eliminar_stop_words)

            # Guardar el contenido preprocesado en el modelo
            documento.contenido = df['texto'].to_string(index=False)
            documento.save()

            return redirect('home')  # Redirige a la página principal después de cargar

    else:
        form = DocumentForm()

    return render(request, 'cargar_documentos.html', {'form': form})

def ver_documentos(request):
    documentos = Documento.objects.all()
    return render(request, 'ver_documentos.html', {'documentos': documentos})

