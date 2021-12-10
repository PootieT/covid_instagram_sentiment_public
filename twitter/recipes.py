import pytorch_lightning as pl

from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertForSequenceClassification
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

import torch


class Roberta(pl.LightningModule):
    def __init__(self, checkpoint: str, num_labels: int, trainData: DataLoader, valData: DataLoader) -> None:
        super(Roberta, self).__init__()

        self.model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels)
        
        self.train_l = trainData
        self.val_l = valData



    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_id):
        outputs = self(**batch)

        loss = outputs.loss
        
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_id):
        outputs = self(**batch)
        loss = outputs.loss
        
        self.log("val_loss", loss)
        return {"val_loss":loss}

       
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        
        self.log('avg_val_loss', avg_loss)
        

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)


    def train_dataloader(self):
        return self.train_l

    def val_dataloader(self):
        return self.val_l

