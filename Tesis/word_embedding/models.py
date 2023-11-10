from django.db import models
from django.contrib.auth.models import User

# Modelo para los documentos
class Document(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    document = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Documento de {self.user.username} ({self.uploaded_at})'

# PAGINA WEB
# Barra de navegacion lateral (carga, etc)
# Restricciones, permisos (tipo de documento, mensajes, etc, errores)
# 1 CARGA
# 2 PREPROSESAMIENTO
# 3 INGRESO DE PARAMETROS (WORD2VEC, DOC2VEC, LEYENDA)
# DATO EXPLORATORIO: BUSCAR PAGINA DE REFERENCIA (REGISTRO DE PROPIEDAD)