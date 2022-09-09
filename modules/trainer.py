from distutils.command.config import config
import json
import torch
import sys

class Trainer():
    def __init__(self, configs):
        super(Trainer, self).__init__()

        self.epoch = configs.trainer.epoch
        self.batch_size = configs.trainer.batch_size
        self.data_parallel = configs.trainer.data_parallel
        self.data_config = configs.data
        self.config = configs
    
    def train(self, train_loader, 
              val_loader, model, criterion,
              optimizer,
              device):
              
        model = model.to(device)

        for batch in train_loader:
            x = batch['x'].to(device)
            p = batch['p'].to(device)
            out = model(x, p)
            print(out.shape)

    
    def inference(self):
        pass