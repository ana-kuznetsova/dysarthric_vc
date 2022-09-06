import torch.utils.data as data
import torch
import torch.nn as nn
import os
import numpy as np
import librosa
import parselmouth
import json
from attrdict import AttrDict



def load_config(path):
    with open(path, 'r') as fo:
        config = json.load(fo)
    config = AttrDict(config)
    return config


def collect_fnames(path):
    all_files = []
    for root, dir, files in os.walk(path):
        for f in files:
            if '.wav' in f:
                f = os.path.join(root, f)
                all_files.append(f)
    return all_files


def get_pitch(path):
    snd = parselmouth.Sound(path)
    pitch = snd.to_pitch()
    pitch_arr = []
    for i in range(pitch.n_frames):
        pitch_arr.append(pitch.get_value_in_frame(i))
    return np.nan_to_num(np.array(pitch_arr))



class LibriTTSData(data.Dataset):
    def __init__(self, configs, mode='train'):
        self.mode = mode
        self.data_path = configs.dataset_path
        self.train_path = os.path.join(self.data_path, configs.train_partition)
        self.test_path = os.path.join(self.data_path, configs.test_partition)
        self.train_files = collect_fnames(self.train_path)
        self.test_files = collect_fnames(self.test_path)


    def __len__(self):
        if self.mode=='train':
            return len(self.train_files)
        return len(self.test_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode=='train':
            return self.train_files[idx]
        return self.test_files[idx]



def collate_fn(data):
    '''
    For batch
    '''
    mel_specs = []
    pitch = []

    for fname in data:
        wav, fs = librosa.load(fname, sr=16000)
        spec = librosa.feature.melspectrogram(y=wav, sr=fs, n_mels=80)
        mel_specs.append(torch.Tensor(spec))
        f0 = get_pitch(fname)
        pitch.append(torch.Tensor(f0))
    
    maxlen_mel = max([i.shape[-1] for i in mel_specs])
    padded_mels = [nn.ZeroPad2d(padding=(0, maxlen_mel - i.shape[-1], 0, 0))(i) for i in mel_specs]
    
    maxlen_p = max([i.shape[-1] for i in pitch])

    padded_pitch = [nn.ConstantPad1d((0, maxlen_p-i.shape[-1]), 0)(i) for i in pitch]

    return {"x":torch.stack(padded_mels), "p": torch.stack(padded_pitch)}