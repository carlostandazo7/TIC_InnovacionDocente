# templatetags/preprocesamiento_tags.py
from django import template
import re
import spacy

register = template.Library()
nlp = spacy.load('es_core_news_sm')

@register.filter(name='preprocesar')
def preprocesar(texto):
    texto = texto.lower()
    texto = re.sub(r'&nbsp;', '', texto)
    texto = texto.lstrip()
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    
    doc = nlp(texto)
    palabras_filtradas = [token.text for token in doc if not token.is_stop]
    
    return ' '.join(palabras_filtradas)
