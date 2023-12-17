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
from sklearn.manifold import TSNE
import numpy as np
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from django.http import HttpResponse
from io import BytesIO
from openpyxl import Workbook


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

def ver_texto_completo(request, documento_id):
    documento = get_object_or_404(Documento, id=documento_id)
    linea_seleccionada = request.GET.get('linea', None)
    
    # Si se proporciona un índice de línea, mostrar solo esa línea
    if linea_seleccionada is not None:
        lineas = documento.contenido_preprocesado.splitlines()
        try:
            linea_seleccionada = int(linea_seleccionada)
            contenido_linea = lineas[linea_seleccionada]
        except (ValueError, IndexError):
            contenido_linea = "Línea no válida"
    else:
        # Si no se proporciona un índice de línea, mostrar todo el contenido
        contenido_linea = "\n".join(documento.contenido_preprocesado.splitlines())

    return render(request, 'ver_texto_completo.html', {'documento': documento, 'contenido_linea': contenido_linea})

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
    
    # Personalizar el modelo

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

    # Si se envió un formulario con una operación, realizarla
    if request.method == 'POST':
        word1 = request.POST.get('word1', '')
        operator = request.POST.get('operator', '')
        word2 = request.POST.get('word2', '')
        result = perform_arithmetic_operation(model, word1, operator, word2)
        context['result'] = result

    return render(request, 'word2vec.html', context)


def perform_arithmetic_operation(model, word1, operator, word2):
    # Verificar que la entrada sea válida
    if not word1 or not operator or not word2:
        return "Entrada no válida. Debe haber dos palabras y un operador."

    if operator not in ['+', '-', '*', '/']:
        return "Operador no válido. Use +, -, *, or /."

    # Realizar la operación
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
            return "Error: " + result_vector
        else:
            similar_words = model.wv.similar_by_vector(result_vector, topn=10)
            result_table = generate_result_table(similar_words)
            return result_table

    else:
        return "Algunas palabras no están en el modelo Word2Vec."


def generate_result_table(similar_words):
    # Generar una tabla HTML con los resultados
    table_html = "<table class='table table-bordered table-hover'><thead><tr><th>Palabra</th><th>Similitud</th></tr></thead><tbody>"
    for word, similarity in similar_words:
        table_html += f"<tr><td>{word}</td><td>{similarity:.4f}</td></tr>"
    table_html += "</tbody></table>"
    return table_html

def word2vec_personalizable(request):
    if request.method == 'POST':
        word = request.POST.get('word', '')
        top_n = int(request.POST.get('top_n', ''))
        if word and top_n:
            similar_words_plot = plot_similar_words(word, top_n)
            return render(request, 'modelo_personalizable.html', {'similar_words_plot': similar_words_plot})
    
    return render(request, 'modelo_personalizable.html')

def plot_similar_words(word, top_n=50):
    # Aquí deberías agregar la lógica para entrenar el modelo con los datos "contenido_preprocesado"
    # y luego obtener las palabras similares y sus vectores para visualizarlos con Plotly
    # El código a continuación es un ejemplo basado en el código proporcionado anteriormente
    # Asegúrate de adaptarlo según tus necesidades

    # Ejemplo de código (puedes modificarlo según tus necesidades)
    documentos = Documento.objects.all()
    documentos_preprocesados = [documento.contenido_preprocesado.lower().split() for documento in documentos]

    bigram = Phrases(documentos_preprocesados, min_count=1, threshold=1)
    trigram = Phrases(bigram[documentos_preprocesados], min_count=1, threshold=1)
    documentos_preprocesados = [trigram[bigram[documento]] for documento in documentos_preprocesados]

    model = Word2Vec(documentos_preprocesados, vector_size=100, window=5, min_count=1, workers=4, max_final_vocab=None)

    similar_words = [sim_word for sim_word, _ in model.wv.most_similar(word, topn=top_n) if sim_word in model.wv]
    similar_words = similar_words[:top_n]
    similar_words.append(word)

    vectors_to_plot = []
    for w in similar_words:
        if w in model.wv:
            vectors_to_plot.append(model.wv[w])

    vectors_to_plot = np.array(vectors_to_plot)

    tsne = TSNE(n_components=2, perplexity=len(similar_words) - 1)
    word_vectors_2d = tsne.fit_transform(vectors_to_plot)

    df = pd.DataFrame(word_vectors_2d, columns=['Dimensión 1', 'Dimensión 2'])
    df['Palabra'] = similar_words

    # Crear una figura similar a plot_word_vectors_zoomable
    fig = go.Figure()

    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Dimensión 1']],
            y=[row['Dimensión 2']],
            text=[row['Palabra']],
            mode='markers',
            name=row['Palabra']
        ))

    fig.update_layout(
        title=f'Visualización de las {top_n} palabras más similares a "{word}"',
        xaxis_title='Dimensión 1',
        yaxis_title='Dimensión 2',
        showlegend=False
    )

    # Habilitar zoom y panorámica
    fig.update_xaxes(type='linear')
    fig.update_yaxes(type='linear')

    # Convertir la figura de Plotly a HTML
    plot_div = fig.to_html(full_html=False)
    return plot_div

# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Preprocesar el texto y lemmatizar usando spaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Función auxiliar para crear tagged_data
def create_tagged_data(documentos):
    documentos_preprocesados = [preprocess_text(documento.contenido_preprocesado.lower()) for documento in documentos]
    return [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(documentos_preprocesados)]

# Definir la función search_similar fuera de la vista
def search_similar(model, documentos, query):
    query_tokens = preprocess_text(query)
    similar_documents = model.dv.most_similar(positive=[model.infer_vector(query_tokens)], topn=5)

    # Filtrar resultados con similitud mayor o igual a 0.5
    filtered_documents = [(int(doc_id), similarity) for doc_id, similarity in similar_documents if similarity >= 0.5]

    # Crear una lista de resultados
    results = []
    for doc_id, similarity in filtered_documents:
        document = documentos[doc_id]  # Obtener el documento correspondiente
        results.append({
            'id': document.id,
            'contenido': document.contenido_preprocesado,
            'similitud': similarity
        })

    return results

def doc2vec(request):
    # Obtener los documentos
    documentos = Documento.objects.all()
    
    # Crear tagged_data
    tagged_data = create_tagged_data(documentos)

    # Definir y entrenar el modelo Doc2Vec
    model = Doc2Vec(vector_size=100, window=5, min_count=1, dm=1, epochs=20, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Si se envió un formulario con una consulta, realizar la búsqueda
    if request.method == 'POST':
        query = request.POST.get('query', '')
        results = search_similar(model, documentos, query)

        # Verificar si no hay resultados
        no_results_message = "No hay similitudes" if not results else None
    else:
        results = None
        no_results_message = None

    context = {
        'documentos': documentos,
        'results': results,
        'no_results_message': no_results_message,
    }

    return render(request, 'doc2vec.html', context)

# Función para descargar resultados
def download_results(request):
    # Obtener los documentos
    documentos = Documento.objects.all()

    # Si se envió un formulario con una consulta, realizar la búsqueda
    query = request.POST.get('query', '')

    # Crear tagged_data
    tagged_data = create_tagged_data(documentos)

    # Crear y entrenar el modelo Doc2Vec
    model = Doc2Vec(vector_size=100, window=5, min_count=1, dm=1, epochs=20, workers=4)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    results = search_similar(model, documentos, query)

    # Crear un DataFrame de Pandas directamente desde los resultados
    df = pd.DataFrame(results)

    # Crear una respuesta HTTP con el archivo XLSX utilizando openpyxl
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename=resultados.xlsx'

    # Crear el archivo Excel utilizando openpyxl
    workbook = Workbook()
    worksheet = workbook.active

    # Agregar resultados
    for _, row in df.iterrows():
        row_data = [row['id'], row['contenido'].lstrip().rstrip(), row['similitud']]
        worksheet.append(row_data)

    # Agregar encabezados
    headers = ['ID', 'Contenido', 'Similitud']
    worksheet.append(headers)

    # Guardar el archivo Excel en la respuesta
    workbook.save(response)

    return response