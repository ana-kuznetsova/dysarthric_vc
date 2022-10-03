import torch
import os


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