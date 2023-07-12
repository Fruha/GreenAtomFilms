from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from torch import nn
import lightning.pytorch as pl
from torchmetrics.functional.classification import binary_auroc, binary_f1_score
from torchmetrics.functional.regression import mean_squared_error
from transformers import DistilBertTokenizerFast
from lightning.pytorch import seed_everything
import torch
from transformers import AutoTokenizer
from nnmodel import ModelBert

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, marks):
        self.encodings = encodings
        self.labels = labels
        self.marks = marks

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        item['marks'] = torch.tensor(self.marks[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)



hparams = {
    'batch_size_train': 16,
    'batch_size_val': 16,
    'global_seed':69,
    'learning_rate': 5e-5,
#     'loss_f1' : nn.MSELoss(),
#     'loss_f2' : nn.MSELoss(),
    'max_epochs' : 3,
}

loader_kwargs = {
    'num_workers': 1, 
#     'pin_memory': True
}

import numpy as np
import pandas as pd
import os
from glob import glob
import re
from tqdm.notebook import tqdm

columns = ['id', 'mark', 'text']

train = []
for i1, file_path in enumerate(tqdm(glob(r'aclImdb/train/pos/*_*.txt') + glob(r'aclImdb/train/neg/*_*.txt'))):
    id_, mark = re.findall(r'aclImdb/train/.*/([0-9]+)_([0-9]+).txt', file_path)[0]
    with open(file_path, 'r', encoding="utf-8") as file:
        text = file.read()
    train.append([int(id_), int(mark), text])
train_df = pd.DataFrame(train, columns=columns)
train_df['label'] = train_df['mark'] >= 6
train_df['text'] = train_df['text'].str.slice(0,2000)

test = []
for i1, file_path in enumerate(tqdm(glob(r'aclImdb/test/pos/*_*.txt') + glob(r'aclImdb/test/neg/*_*.txt'))):
    id_, mark = re.findall(r'aclImdb/test/.*/([0-9]+)_([0-9]+).txt', file_path)[0]
    with open(file_path, 'r', encoding="utf-8") as file:
        text = file.read()
    test.append([int(id_), int(mark), text])
test_df = pd.DataFrame(test, columns=columns)
test_df['label'] = test_df['mark'] >= 6
test_df['text'] = test_df['text'].str.slice(0,2000)


seed_everything(hparams['global_seed'])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

train_encodings = tokenizer(train_df['text'].values.tolist(), truncation=True, padding=True)
val_encodings = tokenizer(test_df['text'].values.tolist(), truncation=True, padding=True)

train_dataset = IMDbDataset(train_encodings, train_df['label'].values.tolist(), train_df['mark'].values.tolist())
val_dataset = IMDbDataset(val_encodings, test_df['label'].values.tolist(), test_df['mark'].values.tolist())

train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size_train'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size_val'], shuffle=True)

bert = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model = ModelBert(bert,hparams)

trainer = pl.Trainer(max_epochs=hparams['max_epochs'])#, limit_train_batches=1000, limit_val_batches=1000)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)