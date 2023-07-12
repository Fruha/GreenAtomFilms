from django.apps import AppConfig
from nnmodel import nnmodel
from transformers import DistilBertTokenizerFast
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from torch import nn
import lightning.pytorch as pl
import torch
from transformers import AutoTokenizer
from pprint import pprint

class PredictModel():
    def __init__(self):
        from django.conf import settings

        print('Created Model')
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        # self.model = nnmodel.ModelBert.load_from_checkpoint(settings.STATIC_ROOT/"filmsai/model.ckpt", model=bert, hparams={})
        self.model = nnmodel.ModelBert.load_from_checkpoint(settings.BASE_DIR/"filmsai/static/filmsai/model.ckpt", model=bert, hparams={})
        self.model.to('cpu')

    def predict(self, texts):
        with torch.no_grad():
            encoding = self.tokenizer(texts, return_tensors='pt', truncation=True, padding=True)
            encoding['input_ids'] = encoding['input_ids'].to(self.model.device)
            encoding['attention_mask'] = encoding['attention_mask'].to(self.model.device)
            outputs = self.model(encoding)
            outputs['labels'] = outputs['labels'].detach().cpu().numpy()
            outputs['marks'] = outputs['marks'].detach().cpu().numpy()
        return outputs

class FilmsaiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'filmsai'
    
    # Здесь можно загрузить что-то до старта сервера
    # https://docs.djangoproject.com/en/4.2/ref/applications/#django.apps.AppConfig.ready
    

    def ready(self):
        self.model = PredictModel()
        # importing model classes
        # from .models import MyModel  # or...
        # MyModel = self.get_model("MyModel")