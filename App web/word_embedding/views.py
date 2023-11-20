from .models import Documento  # Asegúrate de importar tu modelo Documento
from django.contrib.auth.forms import UserCreationForm
from django.urls import reverse_lazy, reverse
from django.views import generic
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django import template
from .forms import *
from django.contrib import messages
from django.db.models import F
import pandas as pd
import spacy
from django.db import transaction
import io
import re
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy
from plotly.offline import plot
import plotly.express as px
import base64

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

# vista para editar contenido

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

# Vista para eliminar contenido

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

# Vista para el preprocesamiento de los datos

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

# Vista para el algoritmo WORD2VEC

# Revisar si es posible descargar el modelo word2vec

def word2vec(request):
    import matplotlib
    matplotlib.use('Agg')
    documentos = Documento.objects.all()

    # Obtener los documentos preprocesados
    documentos_preprocesados = [documento.contenido_preprocesado.lower().split() for documento in documentos]

    # Generar bigramas y trigramas
    bigram = Phrases(documentos_preprocesados, min_count=1, threshold=1)
    trigram = Phrases(bigram[documentos_preprocesados], min_count=1, threshold=1)
    documentos_preprocesados = [trigram[bigram[documento]] for documento in documentos_preprocesados]

    # Entrenar el modelo Word2Vec
    model = Word2Vec(documentos_preprocesados, vector_size=100, window=5, min_count=1, workers=4, max_final_vocab=None)

    # Obtener todas las palabras en el modelo Word2Vec
    words = list(model.wv.index_to_key)

    # Obtener los vectores de palabras
    vectors_to_plot = [model.wv[word] for word in words]

    # Reducción de dimensionalidad con PCA
    pca = PCA(n_components=2)
    word_vectors_2d = pca.fit_transform(vectors_to_plot)

    # Crear un DataFrame con los datos
    df = pd.DataFrame(word_vectors_2d, columns=['Dimensión 1', 'Dimensión 2'])
    df['Palabra'] = words

    # Crear una figura interactiva de Plotly
    fig = go.Figure()

    # Agregar puntos de datos para cada palabra
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(x=[row['Dimensión 1']], y=[row['Dimensión 2']],
                                text=[row['Palabra']], mode='markers', name=row['Palabra']))

    # Configurar el diseño de la figura
    fig.update_layout(
        title='Visualización de Vectores de Palabras',
        xaxis_title='Dimensión 1',
        yaxis_title='Dimensión 2',
        showlegend=False,
    )

    # Convertir la figura de Plotly a HTML
    plot_div = fig.to_html(full_html=False)

    # Generar y mostrar la nube de palabras
    wordcloud = WordCloud(background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 10), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # Convertir la figura de Matplotlib a HTML
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    wordcloud_image = base64.b64encode(buf.getvalue()).decode('utf-8')

    # Incluir las variables necesarias en el contexto
    context = {
        'documentos': documentos,
        'plot_div': plot_div,
        'wordcloud_image': wordcloud_image,
        'words': words,
        'vectors': word_vectors_2d,
        'dimensions': 2,  # Esto puede cambiar según tus necesidades
    }

    return render(request, 'word2vec.html', context)

def operaciones(request):
    if request.method == 'POST':
        # Verifica que 'model' esté definido en tu contexto
        if 'model' not in locals():
            return render(request, 'error.html', {'error_message': 'El modelo Word2Vec no está disponible.'})

        user_input = request.POST.get('arithmetic_operation', '')
        words = user_input.split()

        if len(words) != 3:
            result = "Entrada no válida. Deben ser dos palabras y un operador."
        else:
            word1, operator, word2 = words
            if operator not in ['+', '-', '*', '/']:
                result = "Operador no válido. Use +, -, *, or /."
            else:
                if word1 in model.wv and word2 in model.wv:
                    if operator == '+':
                        result_vector = model.wv[word1] + model.wv[word2]
                    elif operator == '-':
                        result_vector = model.wv[word1] - model.wv[word2]
                    elif operator == '*':
                        result_vector = model.wv[word1] * model.wv[word2]
                    elif operator == '/':
                        result_vector = model.wv[word1] / model.wv[word2] if model.wv[word2].any() != 0 else "División por cero"

                    if isinstance(result_vector, str):
                        result = "Error: " + result_vector
                    else:
                        similar_words = model.wv.similar_by_vector(result_vector, topn=10)
                        result = f"Resultado de la operación: {word1} {operator} {word2} = {similar_words}"
                else:
                    result = "Algunas palabras no están en el modelo Word2Vec."

    else:
        result = None

    context = {'result': result}
    return render(request, 'operaciones.html', context)

# Vista en doc2vec 
# 1. Descargar los resultados
# 2. Agregar botones para ver "mas similares" y "menos similares"
