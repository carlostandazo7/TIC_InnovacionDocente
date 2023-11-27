from django.db import models
from django.contrib.auth.models import User

# Modelo para los documentos
class Documento(models.Model):
    archivo = models.FileField(upload_to='media/archivos_xslx/')
    contenido = models.TextField(blank=True)
    contenido_preprocesado = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.archivo.name