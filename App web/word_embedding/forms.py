from django import forms
from crispy_forms.helper import FormHelper
from crispy_forms.layout import Submit
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
    
    def __init__(self, *args, **kwargs):
        super(DocumentForm, self).__init__(*args, **kwargs)
        self.helper = FormHelper()
        self.helper.form_method = 'post'
        self.helper.add_input(Submit('submit', 'Guardar Cambios', css_class='btn btn-primary'))
        
class DocumentoForm(forms.ModelForm):
    class Meta:
        model = Documento
        fields = ['contenido']
        
class EliminarPalabrasForm(forms.Form):
    palabras_a_eliminar = forms.CharField(
        label='Palabras a eliminar (separadas por comas)',
        widget=forms.TextInput(attrs={'class': 'form-control'}),
        required=False
    )