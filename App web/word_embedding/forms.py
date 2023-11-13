from django import forms
from .models import Documento

class DocumentForm(forms.ModelForm):
    class Meta:
        model = Documento
        fields = ['archivo']
        
    def clean_archivo(self):
        archivo = self.cleaned_data.get('archivo')

        # Verificar si el archivo está vacío
        if archivo and not archivo.read(1):
            raise forms.ValidationError('El archivo está vacío.')

        return archivo