import torch
import os

from TTS.encoder.models.resnet import ResNetSpeakerEncoder
from modules.encoder import GeneralEncoder
from modules.decoder import Tacotron2Conditional


def find(files, num, obj):
    for f in files:
        if num in f and obj in f:
            return f

def restore(conf, model, optimizer=None, scheduler=None, mode='train'):
    path = conf.runner.ckpt_path
    files = os.listdir(path)

    if mode=='train':
        latest = sorted([int(f.split('_')[-1].replace('.pth', '')) for f in files if "_" in f])[-1]
        latest = str(latest)
        print(f"> Restoring from epoch {latest}...")

        model_path = os.path.join(path, find(files, latest, "model"))
        opt_path = os.path.join(path, find(files, latest, "optimizer"))
        if scheduler:
            sched_path = os.path.join(path, find(files, latest, "scheduler"))

        
        model = model.load_state_dict(torch.load(model_path))
        optimizer = optimizer.load_state_dict(torch.load(opt_path))
        if scheduler:
            scheduler = scheduler.load_state_dict(torch.load(sched_path))
    
    else:
        model_path = os.path.join(path, f"best_model.pth")
        model = model.load_state_dict(torch.load(model_path))


def move_device(model_weights):
    '''
    Removes module prefix to load DatParallel model on single device
    '''

    new_weights = {}

    for k in model_weights:
        new_k = k.replace("module.", '')
        new_weights[new_k] = model_weights[k]
    return new_weights

def freeze_params(model, layers=None):
    if not layers:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for layer in layers:
            for name, param in model.named_parameters():
                if layer not in name:
                    param.requires_grad = False


def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def shuffle_tensor(x, dim):
    idx = torch.randperm(x.size(dim))
    x = x[:, idx]
    return x

def init_encoder(config):
    '''
    Initializes Encoder with ResNet as feature extractor
    '''
    
    #Initialize ResNet as feat extractor
    feat_extractor = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)

    #If starting from epoch=0
    if not config.encoder.restore_epoch and config.encoder.spk_enc_path:
        spk_enc_path = os.path.join(config.encoder.spk_enc_path, 'best_model.pth')
        spk_enc_weights = torch.load(spk_enc_path)
        spk_enc_weights = move_device(spk_enc_weights)
        feat_extractor.load_state_dict(spk_enc_weights)

        if config.encoder.freeze_spk_enc and config.encoder.unfreeze_layers:
            freeze_params(feat_extractor, config.model.unfreeze_layers)
        elif config.encoder.freeze_spk_enc:
            freeze_params(feat_extractor)

    #Initialize General encoder

    model = GeneralEncoder(
                feature_extractor=feat_extractor,
                feat_extractor_dim=config.encoder.feat_encoder_dim,
                hidden_dim=config.encoder.hidden_dim, 
                batch_size=config.trainer.batch_size, num_classes=config.data.num_speakers, mi=config.encoder.use_mi)

    if config.encoder.restore_epoch:
        opt_path = os.path.join(config.encoder.ckpt_path, f"optimizer_{config.encoder.restore_epoch}.pth")
        optimizer.load_state_dict(torch.load(opt_path))
        model_path = os.path.join(config.encoder.ckpt_path, f"model_{config.encoder.restore_epoch}.pth")

        ##Avoid pytorch bug from parallel training
        state_dict = torch.load(model_path)
        state_dict_new = {}

        for k in state_dict:
            k_new = k.replace("module.", '')
            state_dict_new[k_new] = state_dict[k]
        state_dict = state_dict_new
        del state_dict_new
        model.load_state_dict(state_dict)
        print(f"Restarted previous experiment from epoch {config.runner.restore_epoch}")
    return model

def init_decoder(config):
    decoder = Tacotron2Conditional()
    decoder.load_state_dict(torch.load(config.decoder.ckpt))
    
    #Freeze encoder, unfreeze decoder and postnet
    
    for name, param in decoder.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return decoder