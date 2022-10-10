from utils.data import LibriTTSData, collate_fn, collate_spk_enc, load_config, VCTKData, VCTKAngleProtoData, collate_spk_enc_augment, UASpeechData, collate_spk_enc_vc
from modules.encoder import GeneralEncoder
from modules.trainer import Trainer
from modules.losses import LossGeneral
from modules.model import JointVC
from utils.eval import evaluate
from utils.utils import move_device, freeze_params, get_features, init_encoder, init_decoder

from torch.utils.data import random_split, DataLoader
from TTS.encoder.models.resnet import ResNetSpeakerEncoder
from TTS.encoder.losses import AngleProtoLoss, SoftmaxAngleProtoLoss
from torch.utils.data import Subset
import torch

import wandb
import sys
import json
import argparse
import os


def run_training(config, config_path):
    torch.cuda.empty_cache()

    #wandb configs
    if config.runner.wandb:
        wandb.init(config=config)
        #wandb.config(project=config.runner.project_name, entity=config.runner.entity)
        wandb.project = config.runner.project_name
        wandb.entity = config.runner.entity
        wandb.run.name = config.runner.run_name
        print(f"> Initialized run with run name {config.runner.run_name}")
    if config.runner.wandb and config.runner.log_config:
        with open(config_path, 'r') as fo:
            config_json = json.load(fo)
        wandb.config.update = config_json

    ##Preload data
    if config.data.dataset=='LibriTTS':
        dataset = LibriTTSData(config, mode='train')

    elif (config.data.dataset=='VCTK' or config.data.dataset=='DysarthricSim') and config.model.model_name=='speaker_encoder':
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

        if config.data.augment:
            collate_fn_enc = collate_spk_enc_augment
        else:
            collate_fn_enc = collate_spk_enc

        train_loader = DataLoader(train,
                                batch_size=config.trainer.batch_size, 
                                shuffle=False, collate_fn=collate_fn_enc,
                                drop_last=True, num_workers=2, pin_memory=False
                            )
        val_loader = DataLoader(val,
                                    batch_size=config.trainer.batch_size, 
                                    shuffle=False, collate_fn=collate_spk_enc,
                                    drop_last=True, num_workers=2, pin_memory=False
                                )

    else:

        train_len = int(len(dataset)*0.9)
        val_len = len(dataset) -  int(len(dataset)*0.9)

        train, val = random_split(dataset, [train_len, val_len], 
                                    generator=torch.Generator().manual_seed(42))

        if config.model.model_name=='speaker_encoder':
            if config.data.augment:
                train_collate = collate_spk_enc_augment
            else:
                train_collate = collate_spk_enc
            val_collate = collate_spk_enc
        elif config.model.model_name=='joint_vc':
            train_collate = collate_spk_enc_vc
            val_collate = collate_spk_enc_vc

        train_loader = DataLoader(train,
                                    batch_size=config.trainer.batch_size, 
                                    shuffle=True, collate_fn=train_collate,
                                    drop_last=True, num_workers=2, pin_memory=False
                                )
        val_loader = DataLoader(val,
                                    batch_size=config.trainer.batch_size, 
                                    shuffle=True, collate_fn=val_collate,
                                    drop_last=True, num_workers=2, pin_memory=False
                                )


    ## Define model
    avail_models = ["speaker_encoder", "general_encoder", "joint_vc"]

    if not config.model.model_name in avail_models:
        raise NotImplementedError(f"{config.model.model_name}: model not implemented")

    #Define model, riterion and optimizer for each model

    if config.model.model_name=='speaker_encoder':
        model = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)
        ###DOTO: Make a function for restoring/freezing params
        if not config.runner.restore_epoch and config.runner.spk_enc_path:
            spk_enc_path = os.path.join(config.runner.spk_enc_path, 'best_model.pth')
            spk_enc_weights = torch.load(spk_enc_path)
            spk_enc_weights = move_device(spk_enc_weights)
            model.load_state_dict(spk_enc_weights)
            if config.model.freeze_spk_enc and config.model.unfreeze_layers:
                freeze_params(model, config.model.unfreeze_layers)

        criterion = SoftmaxAngleProtoLoss(embedding_dim=config.model.feat_encoder_dim, n_speakers=config.data.num_speakers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-5)
        if config.trainer.scheduler:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.25)
        else:
            lr_scheduler = None
    
    elif config.model.model_name == 'joint_vc':
        #Initialize encoder
        encoder = init_encoder(config)
        decoder = init_decoder(config)
        model = JointVC(encoder, decoder)
        ## Initialize attention scheduler if needed
        criterion = LossGeneral(config.loss.alpha1, config.loss.alpha2, config.loss.alpha3)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.trainer.lr)
        lr_scheduler = None


    trainer = Trainer(config)
    trainer.train(train_loader, val_loader, 
                  model, criterion,
                  optimizer, lr_scheduler, 
                  device=torch.device("cuda:0"), parallel=config.runner.data_parallel)

def run_inference(config, out_dir):

    if config.data.dataset=='VCTK' and config.model.model_name=='speaker_encoder':
        dataset_test = VCTKAngleProtoData(config, mode='test')

    elif config.data.dataset=='UASpeech':
        dataset_test = UASpeechData(config)

    test_loader = DataLoader(dataset_test, batch_size=config.trainer.batch_size, 
                                shuffle=False, collate_fn=collate_spk_enc,
                                drop_last=True, num_workers=2, pin_memory=False
                            )

    model = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)

    trainer = Trainer(config)
    trainer.inference(test_loader, model, 
                        device=torch.device("cuda:0"), 
                        parallel=config.runner.data_parallel,
                        out_dir=out_dir)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help='Path to config .json')
    parser.add_argument("-m", "--mode", type=str, help='Train, inference, test', required=True)
    parser.add_argument("-o", "--out", help="Path to the inference output")
    parser.add_argument("-i", "--inp", help="Path to the input dir")
    args = parser.parse_args()


    config = load_config(args.config)

    if args.mode=='train':
        run_training(config, args.config)

    if args.mode=='inference':
        assert args.out!=None
        run_inference(config, args.out)

    if args.mode=='test':
        assert args.inp!=None
        evaluate(args.inp, 'speaker_encoder')