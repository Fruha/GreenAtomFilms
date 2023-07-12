from django import forms
from filmsai import models


class FilmsBaseForm(forms.ModelForm):
    class Meta:
        model = models.Films
        fields = '__all__'

class CommentCreateForm(forms.ModelForm):
    class Meta:
        model = models.Comment
        # fields = ['film','comment']
        exclude = ['rating','assesment']
        widgets= {
            'film': forms.HiddenInput()
        }