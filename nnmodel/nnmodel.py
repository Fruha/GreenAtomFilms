from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
from torch import nn
import lightning.pytorch as pl
from torchmetrics.functional.classification import binary_auroc, binary_f1_score
from torchmetrics.functional.regression import mean_squared_error
from transformers import DistilBertTokenizerFast
import torch
from transformers import AutoTokenizer

class ModelBert(pl.LightningModule):
    def __init__(self, model, hparams:dict):
        super(ModelBert, self).__init__()
        self.model = model
        self.loss_f1 = nn.MSELoss()
        self.loss_f2 = nn.MSELoss()
        self.hparams.update(hparams)
#         self.save_hyperparameters()
        
        self.train_step_outputs_labels = []
        self.train_step_outputs_marks = []
        self.train_step_target_labels = []
        self.train_step_target_marks = []
        self.val_step_outputs_labels = []
        self.val_step_outputs_marks = []
        self.val_step_target_labels = []
        self.val_step_target_marks = []
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        self.train_step_outputs_labels.append(outputs['labels'].detach().cpu())
        self.train_step_outputs_marks.append(outputs['marks'].detach().cpu())
        self.train_step_target_labels.append(batch['labels'].detach().cpu())
        self.train_step_target_marks.append(batch['marks'].detach().cpu())
        
        loss1 = self.loss_f1(batch['labels'], outputs['labels'])
        loss2 = self.loss_f2(batch['marks'], outputs['marks'])
        loss = loss1 + loss2
        
        self.log("Loss1/Train", loss1)
        self.log("Loss2/Train", loss2)
        self.log("Loss/Train", loss, prog_bar=True)        
        return loss

    def on_train_epoch_end(self):
        labels_otputs = torch.cat(self.train_step_outputs_labels)
        marks_otputs = torch.cat(self.train_step_outputs_marks)
        labels_target = torch.cat(self.train_step_target_labels)
        marks_target = torch.cat(self.train_step_target_marks)
        self.log("ROCAUC/Train", binary_auroc(labels_otputs, labels_target.to(torch.long)))
        self.log("F1Score/Train", binary_f1_score(labels_otputs, labels_target))
        self.log("RMSE/Train", mean_squared_error(marks_otputs, marks_target, squared=False))
        self.train_step_outputs_labels = []
        self.train_step_outputs_marks = []
        self.train_step_target_labels = []
        self.train_step_target_marks = []
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        self.val_step_outputs_labels.append(outputs['labels'].detach().cpu())
        self.val_step_outputs_marks.append(outputs['marks'].detach().cpu())
        self.val_step_target_labels.append(batch['labels'].detach().cpu())
        self.val_step_target_marks.append(batch['marks'].detach().cpu())
        
        loss1 = self.loss_f1(batch['labels'], outputs['labels'])
        loss2 = self.loss_f2(batch['marks'], outputs['marks'])
        loss = loss1 + loss2
        
        self.log("Loss1/Val", loss1)
        self.log("Loss2/Val", loss2)
        self.log("Loss/Val", loss, prog_bar=True)        
        return loss

    def on_validation_epoch_end(self):
        labels_otputs = torch.cat(self.val_step_outputs_labels)
        marks_otputs = torch.cat(self.val_step_outputs_marks)
        labels_target = torch.cat(self.val_step_target_labels)
        marks_target = torch.cat(self.val_step_target_marks)
        self.log("ROCAUC/Val", binary_auroc(labels_otputs, labels_target.to(torch.long)))
        self.log("F1Score/Val", binary_f1_score(labels_otputs, labels_target))
        self.log("RMSE/Val", mean_squared_error(marks_otputs, marks_target, squared=False))
        self.val_step_outputs_labels = []
        self.val_step_outputs_marks = []
        self.val_step_target_labels = []
        self.val_step_target_marks = []
    
#     def on_save_checkpoint(self, checkpoint):
#         self.logger.experiment.log_model('model', 'models', overwrite=True)
    
    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'])
#         self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.hparams['gamma'])
        return {
            'optimizer': self.optimizer,
#             'lr_scheduler':self.scheduler,
        }
    
    def forward(self, batch):
        outputs = self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        ans = {
            'labels' : torch.nn.functional.sigmoid(outputs.logits[:,0]),
            'marks' : outputs.logits[:,1],
        }
        return ans