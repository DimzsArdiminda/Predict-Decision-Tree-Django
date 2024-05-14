from django import forms
from main.models import Datasets

class DatasetForm(forms.ModelForm):
    class Meta:
        model = Datasets
        fields = ('title', 'dataset')