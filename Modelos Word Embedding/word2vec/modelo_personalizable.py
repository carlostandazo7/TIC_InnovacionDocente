import pandas as pd
import spacy
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from sklearn.manifold import TSNE
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter
import random


# Paso 1: Leer el documento CSV
data = pd.read_csv('E:/UTPL/8C/Practicum 4.1/Primer Bimestre/Modelos Word Embedding/data/proyectos_innovacion_docente.csv', delimiter=',', encoding='unicode_escape')

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

# Función para visualizar los vectores de palabras similares con diseño similar a plot_word_vectors_zoomable
def plot_similar_words(word, top_n=50):
    similar_words = [sim_word for sim_word, _ in model.wv.most_similar(word, topn=top_n) if sim_word in model.wv]
    similar_words = similar_words[:50]
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
        title=f'Visualización de las 50 palabras más similares a "{word}"',
        xaxis_title='Dimensión 1',
        yaxis_title='Dimensión 2',
        showlegend=False
    )

    # Habilitar zoom y panorámica
    fig.update_xaxes(type='linear')
    fig.update_yaxes(type='linear')

    fig.show()

    wordcloud = WordCloud(background_color='white').generate(' '.join(similar_words))
    fig_wordcloud = go.Figure(go.Image(z=wordcloud.to_array()))
    fig_wordcloud.update_layout(title=f'Nube de palabras para palabras similares a "{word}"')
    fig_wordcloud.show()

def plot_word_frequency(data, top_n=20):
    tokens = []
    for text in data['texto']:
        doc = nlp(text.lower())
        tokens.extend([token.text for token in doc if not token.is_punct and not token.is_space])

    word_freq = Counter(tokens)

    most_common_words = word_freq.most_common(top_n)
    words = [word for word, freq in most_common_words]
    frequencies = [freq for word, freq in most_common_words]

    fig = px.bar(x=words, y=frequencies, labels={'x': 'Palabra', 'y': 'Frecuencia'}, text=words)
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        title=f'Las {top_n} palabras más frecuentes',
        xaxis_title='Palabra',
        yaxis_title='Frecuencia',
        xaxis={'categoryorder': 'total descending'}
    )
    fig.show()

# Ejemplo de uso del modelo entrenado
word = 'docente'
plot_similar_words(word, top_n=50)

# Visualizar la frecuencia de las palabras más frecuentes en el data
plot_word_frequency(data)
