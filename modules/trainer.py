from distutils.command.config import config
import json
import torch
import sys

from utils.data import LibriTTSData, collate_fn, load_config, VCTKData
from encoder import DysarthEncoder

from torch.utils.data import random_split, DataLoader


class Trainer():
    def __init__(self, configs):
        super(Trainer, self).__init__()

        self.epoch = configs.trainer.epoch
        self.batch_size = configs.trainer.batch_size
        self.ckpt = configs.trainer.ckpt
        self.data_parallel = configs.trainer.data_parallel
        self.data_config = configs.data
        self.config = configs
    
    def train(self):

        ##Preload data
        if self.config.data.dataset=='LibriTTS':
            dataset = LibriTTSData(self.data_config, mode='train')
        else:
            dataset = VCTKData(self.data_config, mode='train')
        train_len = int(len(dataset)*0.9)
        val_len = len(dataset) -  int(len(dataset)*0.9)

        print(f"len dataset {len(dataset)}")
        train, val = random_split(dataset, [train_len, val_len], 
                                  generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train,
                                  batch_size=self.batch_size, 
                                  shuffle=True, collate_fn=collate_fn,
                                  drop_last=True, num_workers=2
                                )
        val_loader = DataLoader(val,
                                  batch_size=self.batch_size, 
                                  shuffle=True, collate_fn=collate_fn,
                                  drop_last=True, num_workers=2
                                )
        device = "cuda"
        model = DysarthEncoder(self.config)
        model = model.to(device)

        for batch in train_loader:
            x = batch['x'].to(device)
            out = model(x)
            print(out.shape)

    
    def inference(self):
        pass
    


if __name__ == "__main__":
    config = load_config(sys.argv[1])
    trainer = Trainer(config)
    trainer.train()