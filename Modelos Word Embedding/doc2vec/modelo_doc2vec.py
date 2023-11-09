import pandas as pd
import spacy
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import matplotlib.pyplot as plt

# Cargar el modelo de spaCy en español
nlp = spacy.load("es_core_news_sm")

# Cargar el archivo CSV
data = pd.read_csv('E:/UTPL/8C/Practicum 4.1/Primer Bimestre/Modelos Word Embedding/data/proyectos_innovacion_docente.csv', delimiter =',', encoding = 'unicode_escape',)

# Preprocesar el texto y lemmatizar usando spaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return tokens

# Aplicar preprocesamiento a cada fila del DataFrame
df['tokens'] = df['texto'].apply(preprocess_text)

# Crear objetos TaggedDocument para el entrenamiento
tagged_data = [TaggedDocument(words=doc, tags=[str(i)]) for i, doc in enumerate(df['tokens'])]

# Definir y entrenar el modelo Doc2Vec
model = Doc2Vec(vector_size=100, window=5, min_count=1, dm=1, epochs=20, workers=4)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)
model.save("E:/UTPL/8C/Practicum 4.1/Primer Bimestre/Semana 8/modelo_doc2vec.model")

# Función para buscar documentos similares y mostrar resultados en una imagen
def search_similar():
    query = "Proyectos de innovación docente con diseños periodicos"
    query_tokens = preprocess_text(query)
    similar_documents = model.dv.most_similar(positive=[model.infer_vector(query_tokens)], topn=5)

    # Crear una imagen para mostrar los resultados
    plt.figure(figsize=(8, 6))
    plt.barh([f"Documento {doc_id}" for doc_id, _ in similar_documents], [similarity for _, similarity in similar_documents])
    plt.xlabel("Similitud")
    plt.title("Documentos Similares")
    plt.savefig("resultados_doc2vec.png")  # Guardar la imagen como archivo
    plt.show()

# Llamar a la función para buscar documentos similares
search_similar()

# Logueo colores de la U
# carga de documento
# preprocesar
# presentar nube de palabras
# presentar modelos
# mas operaciones aritmeticas
