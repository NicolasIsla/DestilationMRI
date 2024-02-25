import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torch.optim.lr_scheduler as lr_scheduler
from torchmetrics import Accuracy

torch.set_float32_matmul_precision('high')

class TrainerModule(pl.LightningModule):
    def __init__(self, model, learning_rate=0.5, momentum=0.9):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.model.num_classes)
        self.loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.lr = learning_rate
        self.momentum = momentum

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        # Parámetros del optimizador
        lr = 0.5
        lr_warmup_epochs = 5
        weight_decay = 2e-05
        momentum = 0.9

        # No poner weight_decay en las capas de BatchNormalization
        parameters = [
            {'params': [p for n, p in self.model.named_parameters() if 'bn' not in n], 'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if 'bn' in n], 'weight_decay': 0}
        ]
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum)
        # optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        
        final_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        # Agregar warmup al scheduler
        if lr_warmup_epochs > 0:
            warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch + 1) / (lr_warmup_epochs + 1), 1))
        
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, final_scheduler], milestones=[lr_warmup_epochs])
        
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.train_accuracy(logits, y)
        return loss

    def on_training_epoch_end(self, outputs = None):
        self.log('train/accuracy', self.train_accuracy.compute()*100, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        self.val_accuracy(logits, y)

    def on_validation_epoch_end(self, outputs = None):
        self.log('val/accuracy', self.val_accuracy.compute()*100, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model(x)
        loss = self.loss(logits, y)
        self.log('test/loss', loss)
        self.test_accuracy(logits, y)

    def on_test_epoch_end(self, outputs = None):
        self.log('test/accuracy', self.test_accuracy.compute()*100, prog_bar=True, on_epoch=True)
    
    # Agregar learning rate a los logs
    def on_train_epoch_start(self):
        lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', lr, on_epoch=True)
        
if __name__ == '__main__':
    from utils import load_model_student
    import sys
    import argparse

    # Directorio de logs
    log_dir = "trainer_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    