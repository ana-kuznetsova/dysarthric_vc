from utils.data import LibriTTSData, collate_fn, load_config, VCTKData, VCTKAngleProtoData
from modules.encoder import GeneralEncoder
from modules.trainer import Trainer

from torch.utils.data import random_split, DataLoader
from TTS.encoder.models.resnet import ResNetSpeakerEncoder
from TTS.encoder.losses import AngleProtoLoss
from torch.utils.data import Subset
import torch

import wandb
import sys


def run_training(config):

    #wandb configs
    if config.runner.wandb:
        wandb.init(project=config.runner.project_name, entity=config.runner.entity)
    if config.runner.log_config:
        wandb.config = config
        #wandb.log({"loss": loss})

    
    #Cuda settings
    device = config.runner.cuda_device


    ##Preload data
    if config.data.dataset=='LibriTTS':
        dataset = LibriTTSData(config, mode='train')

    elif config.data.dataset=='VCTK' and config.model.model_name=='speaker_encoder':
        dataset = VCTKAngleProtoData(config, mode='train')
    else:
        dataset = VCTKData(config, mode='train')


    if config.model.model_name=='speaker_encoder':
        #Do not shuffle
        num_batches = int(len(dataset)*0.9//config.trainer.batch_size)
        train_len = int(config.trainer.batch_size*num_batches)
        num_batches = int(len(dataset)*0.1//config.trainer.batch_size)
        val_len = int(config.trainer.batch_size*num_batches) 

        train_ind = [i for i in range(train_len)]
        val_ind = [train_len+i for i in range(val_len)]
        
        train = Subset(dataset, train_ind)
        val = Subset(dataset, val_ind)
        print(f"train {len(train)}, val {len(val)}")

        train_loader = DataLoader(train,
                                batch_size=config.trainer.batch_size, 
                                shuffle=False, collate_fn=collate_fn,
                                drop_last=True, num_workers=2
                            )
        val_loader = DataLoader(val,
                                    batch_size=config.trainer.batch_size, 
                                    shuffle=False, collate_fn=collate_fn,
                                    drop_last=True, num_workers=2
                                )

    else:
    
        train_len = int(len(dataset)*0.9)
        val_len = len(dataset) -  int(len(dataset)*0.9)

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


    ## Define model
    avail_models = ["speaker_encoder", "general_encoder", "joint_vc"]

    if not config.model.model_name in avail_models:
        raise NotImplementedError(f"{config.model.model_name}: model not implemented")

    #Define model, riterion and optimizer for each model

    if config.model.model_name=='speaker_encoder':
        model = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)
        criterion = AngleProtoLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.25)

    elif config.model.model_name=='general_encoder':
        feat_extractor = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)
        model = GeneralEncoder(inp_feature_dim=config.model.feat_encoder_dim,
                feature_extractor=feat_extractor,
                feat_extractor_dim=config.model.feat_encoder_dim,
                hidden_dim=config.model.hidden_dim, 
                batch_size=config.trainer.batch_size)

    trainer = Trainer(config)
    trainer.train(train_loader, val_loader, 
                  model, criterion,
                  optimizer, lr_scheduler, 
                  device)

if __name__ == "__main__":
    config = load_config(sys.argv[1])
    run_training(config)