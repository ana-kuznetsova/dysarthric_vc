from utils.data import LibriTTSData, collate_fn, load_config, VCTKData
from modules.encoder import GeneralEncoder
from modules.Trainer import Trainer

from torch.utils.data import random_split, DataLoader
from TTS.encoder.models.resnet import ResNetSpeakerEncoder


def run_training(config):
    ##Preload data
    if config.data.dataset=='LibriTTS':
        dataset = LibriTTSData(self.data_config, mode='train')
    else:
        dataset = VCTKData(self.data_config, mode='train')
    
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

    if config.model.model_name=='speaker_encoder':
        feat_extractor = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)

    elif config.model.model_name=='general_encoder':
        feat_extractor = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)
        model = GeneralEncoder(inp_feature_dim=config.model.feat_encoder_dim,
                feature_extractor=feat_extractor,
                feat_extractor_dim=config.model.feat_encoder_dim,
                hidden_dim=config.model.hidden_dim, 
                batch_size=config.trainer.batch_size)

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    config = load_config(sys.argv[1])