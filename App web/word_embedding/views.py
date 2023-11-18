from .models import Documento  # Asegúrate de importar tu modelo Documento
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy
from django.views import generic
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django import template
from .forms import *
import pandas as pd
import spacy
from django.db import transaction
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

def cargar_documentos(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            documento = form.save(commit=False)

            # Leer el archivo XLSX
            df = pd.read_excel(documento.archivo)

            # Realizar las operaciones de limpieza o procesamiento según sea necesario
            # Puedes personalizar esta parte según tus necesidades
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            # Guardar el contenido preprocesado en el modelo
            documento.contenido = df.to_string(index=False)

            # Guardar el documento en la base de datos
            documento.save()

            return redirect('home')  # Redirige a la página principal después de cargar

    else:
        form = DocumentForm()

    return render(request, 'cargar_documentos.html', {'form': form})

def ver_documentos(request):
    documentos = Documento.objects.all()
    return render(request, 'ver_documentos.html', {'documentos': documentos})

def ver_contenido(request, documento_id):
    documento = get_object_or_404(Documento, id=documento_id)
    contenido_linea = request.GET.get('contenido', '0')

    try:
        indice_linea = int(contenido_linea)
    except ValueError:
        indice_linea = 0

    lineas_contenido = documento.contenido.splitlines()
    if 0 <= indice_linea < len(lineas_contenido):
        linea_seleccionada = lineas_contenido[indice_linea]
    else:
        linea_seleccionada = "Índice de línea no válido"

    return render(request, 'ver_contenido.html', {'linea_seleccionada': linea_seleccionada})

@transaction.atomic
def editar_contenido(request, documento_id):
    documento = get_object_or_404(Documento, id=documento_id)
    contenido_linea = request.GET.get('contenido_linea', '0')

    try:
        indice_linea = int(contenido_linea)
    except ValueError:
        indice_linea = 0

    lineas_contenido = documento.contenido.splitlines()
    if 0 <= indice_linea < len(lineas_contenido):
        linea_seleccionada = lineas_contenido[indice_linea]
    else:
        linea_seleccionada = "Índice de línea no válido"

    if request.method == 'POST':
        nuevo_contenido = request.POST.get('nuevo_contenido', '')

        # Actualiza la línea en la lista
        lineas_contenido[indice_linea] = nuevo_contenido

        # Actualiza el contenido del documento
        documento.contenido = '\n'.join(lineas_contenido)
        documento.save()

        # Redirige a la vista 'ver_documentos' después de la edición
        return redirect('ver_documentos')

    return render(request, 'editar_contenido.html', {'documento': documento, 'contenido_linea': contenido_linea, 'linea_seleccionada': linea_seleccionada})

@require_POST
def eliminar_documento(request, documento_id):
    documento = get_object_or_404(Documento, id=documento_id)
    contenido_linea = request.GET.get('contenido', '0')

    try:
        indice_linea = int(contenido_linea)
    except ValueError:
        indice_linea = 0

    lineas_contenido = documento.contenido.splitlines()

    if 0 <= indice_linea < len(lineas_contenido):
        # Elimina la línea de la lista
        del lineas_contenido[indice_linea]

        # Actualiza el contenido del documento
        documento.contenido = '\n'.join(lineas_contenido)
        documento.save()

    # Redirige a la vista 'ver_documentos' después de eliminar la línea
    return redirect('ver_documentos')

register = template.Library()

@register.filter(name='preprocesar')
def preprocesar(request):
    documentos = Documento.objects.all()

    # Cargar el modelo de spaCy
    nlp = spacy.load('es_core_news_sm')

    form = EliminarPalabrasForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        # Obtener las palabras a eliminar del formulario
        palabras_a_eliminar = form.cleaned_data['palabras_a_eliminar'].split(',')

        # Realizar preprocesamiento para cada documento
        for documento in documentos:
            contenido_preprocesado = documento.contenido.lower()

            for palabra in palabras_a_eliminar:
                contenido_preprocesado = contenido_preprocesado.replace(palabra, '')

            contenido_preprocesado = contenido_preprocesado.lstrip()
            contenido_preprocesado = re.sub(r'\d+', '', contenido_preprocesado)
            contenido_preprocesado = re.sub(r'[^\w\s]', '', contenido_preprocesado)

            # Eliminar stop words utilizando spaCy
            doc = nlp(contenido_preprocesado)
            palabras_filtradas = [token.text for token in doc if not token.is_stop]
            contenido_preprocesado = ' '.join(palabras_filtradas)

            # Actualizar el campo en el modelo
            documento.contenido_preprocesado = contenido_preprocesado
            documento.save()

    context = {'documentos': documentos, 'form': form}
    return render(request, 'preprocesar.html', context)


# Lematizar
# Corrección ortográfica.