from utils.data import LibriTTSData, collate_fn, load_config, VCTKData, VCTKAngleProtoData
from modules.encoder import GeneralEncoder
from modules.Trainer import Trainer

from torch.utils.data import random_split, DataLoader
from TTS.encoder.models.resnet import ResNetSpeakerEncoder
from TTS.encoder.losses import AngleProtoLoss
from torch.utils.data import Subset

import wandb


def run_training(config):

    #wandb configs
    if config.runner.wandb:
        wandb.init(project=config.runner.project_name, entity=config.runner.entity)
    if config.runner.log_configs:
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
        num_batches = len(dataset)*0.9//config.data.batch_size
        train_len = config.data.batch_size*num_batches
        val_len = len(dataset) - train_len 
        train_ind = [i for in in range(train_len)]
        val_ind = [train_len+i for i in range(val_len)]
        
        train = Subset(dataset, train_ind)
        val = Subset(dataset, val_ind)

        train_loader = DataLoader(train,
                                batch_size=self.batch_size, 
                                shuffle=False, collate_fn=collate_fn,
                                drop_last=True, num_workers=2
                            )
        val_loader = DataLoader(val,
                                    batch_size=self.batch_size, 
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
        loss = AngleProtoLoss()

    elif config.model.model_name=='general_encoder':
        feat_extractor = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)
        model = GeneralEncoder(inp_feature_dim=config.model.feat_encoder_dim,
                feature_extractor=feat_extractor,
                feat_extractor_dim=config.model.feat_encoder_dim,
                hidden_dim=config.model.hidden_dim, 
                batch_size=config.trainer.batch_size)

    trainer = Trainer(config)
    trainer.train(train_loader, val_loader, model, criterion, device)

if __name__ == "__main__":
    config = load_config(sys.argv[1])