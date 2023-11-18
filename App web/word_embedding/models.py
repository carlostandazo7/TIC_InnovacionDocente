from django.db import models
from django.contrib.auth.models import User

# Modelo para los documentos
class Documento(models.Model):
    archivo = models.FileField(upload_to='media/archivos_xslx/')
    contenido = models.TextField(blank=True)
    contenido_preprocesado = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return self.archivo.name


# PAGINA WEB
# Barra de navegacion lateral (carga, etc)
# Restricciones, permisos (tipo de documento, mensajes, etc, errores)
# 1 CARGA
# 2 PREPROSESAMIENTO
# 3 INGRESO DE PARAMETROS (WORD2VEC, DOC2VEC, LEYENDA)
# DATO EXPLORATORIO: BUSCAR PAGINA DE REFERENCIA (REGISTRO DE PROPIEDAD)