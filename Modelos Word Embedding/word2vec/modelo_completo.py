import pandas as pd
import spacy
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Paso 1: Leer el documento CSV
data = pd.read_csv('E:/UTPL/8C/Practicum 4.1/Primer Bimestre/Modelos Word Embedding/data/proyectos_innovacion_docente.csv', delimiter =',', encoding = 'unicode_escape',)

# Paso 2: Preprocesar los datos con Spacy
nlp = spacy.load('es_core_news_sm')
sentences = []
for text in data['texto']:
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_punct and not token.is_space]
    sentences.append(tokens)

# Generar bigramas y trigramas
bigram = Phrases(sentences, min_count=1, threshold=1)
trigram = Phrases(bigram[sentences], min_count=1, threshold=1)
sentences = [trigram[bigram[sentence]] for sentence in sentences]

# Paso 3: Entrenar el modelo Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4, max_final_vocab=None)

# Función para visualizar los vectores de palabras con zoom y panorámica
def plot_word_vectors_zoomable():
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

    # Habilitar zoom y panorámica
    fig.update_xaxes(type='linear')
    fig.update_yaxes(type='linear')

    # Mostrar la figura interactiva
    fig.show()

    # Generar y mostrar la nube de palabras
    wordcloud = WordCloud(background_color='white').generate(' '.join(words))
    plt.figure(figsize=(10, 10), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# Función para realizar operaciones aritméticas
def perform_arithmetic_operation():
    print("Realizar operaciones aritméticas con el modelo Word2Vec:")
    print("Ingrese una operacion aritmetica:")
    user_input = input("Operación: ")
    words = user_input.split()

    # Verificar que la entrada sea válida
    if len(words) != 3:
        print("Entrada no válida. Debe haber dos palabras y un operador.")
        return

    word1, operator, word2 = words
    if operator not in ['+', '-', '*', '/']:
        print("Operador no válido. Use +, -, *, or /.")
        return

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
            print("Error: " + result_vector)
        else:
            similar_words = model.wv.similar_by_vector(result_vector, topn=10)
            print(f"Resultado de la operación: {word1} {operator} {word2} = {similar_words}")

    else:
        print("Algunas palabras no están en el modelo Word2Vec.")

# Visualizar el modelo Word2Vec con zoom y panorámica
plot_word_vectors_zoomable()

# Realizar operaciones aritméticas
perform_arithmetic_operation()