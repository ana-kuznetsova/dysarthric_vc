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
            if '.wav' in f or '.flac' in f:
                f = os.path.join(root, f)
                all_files.append(f)
    return all_files


def get_pitch(path):
    snd = parselmouth.Sound(path[0]).resample(16000)
    pitch = snd.to_pitch()
    pitch_arr = []
    for i in range(pitch.n_frames):
        pitch_arr.append(pitch.get_value_in_frame(i))
    return np.nan_to_num(np.array(pitch_arr))


def filter_speakers(configs):
    test_spk = configs.test_partition
    ignore = configs.ignore_speakers
    all_wav_paths = collect_fnames(configs.dataset_path)
    all_wav_paths = [f for f in all_wav_paths for i in ignore if i not in f]
    test_files = []
    train_files = []

    for f in all_wav_paths:
        for t in test_spk:
            if t not in f:
                test_files.append(f)
            else:
                train_files.append(f)

    unique_speakers = os.listdir(configs.dataset_path)
    for i in ignore:
        unique_speakers.remove(i)
        unique_speakers.remove('log.txt')

    spk2id_map = {}

    for i in range(len(unique_speakers)):
        spk2id_map[unique_speakers[i]] = i
    
    print(f"Found {len(spk2id_map.keys())} unique speakers, skipping {ignore}...")

    train_spk_ids = []
    test_spk_ids = []
    
    for i in train_files:
        spk_id = i.split('/')[-1].split('_')[0]
        train_spk_ids.append(spk2id_map[spk_id])

    for i in test_files:
        spk_id = i.split('/')[-1].split('_')[0]
        test_spk_ids.append(spk2id_map[spk_id])

    return train_files, train_spk_ids,  test_files, test_spk_ids

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


class VCTKData(data.Dataset):
    def __init__(self, configs, mode='train'):
        self.mode = mode
        self.data_path = configs.dataset_path
        
        train_files, train_spk_ids, test_files, test_spk_ids = filter_speakers(configs)
        self.train_files = train_files
        self.train_spk_ids = train_spk_ids
        self.test_files = test_files
        self.test_spk_ids = test_spk_ids

    def __len__(self):
        if self.mode=='train':
            return len(self.train_files)
        return len(self.test_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode=='train':
            return (self.train_files[idx], self.train_spk_ids[idx])
        return (self.test_files[idx], self.test_spk_ids[idx])



def collate_fn(data):
    '''
    For batch
    '''
    mel_specs = []
    pitch = []

    for fname in data:
        wav, fs = librosa.load(fname[0], sr=16000)
        spec = librosa.feature.melspectrogram(y=wav, sr=fs, n_mels=80)
        mel_specs.append(torch.Tensor(spec))
        f0 = get_pitch(fname)
        pitch.append(torch.Tensor(f0))
    
    maxlen_mel = max([i.shape[-1] for i in mel_specs])
    padded_mels = [nn.ZeroPad2d(padding=(0, maxlen_mel - i.shape[-1], 0, 0))(i) for i in mel_specs]
    
    maxlen_p = max([i.shape[-1] for i in pitch])

    padded_pitch = [nn.ConstantPad1d((0, maxlen_p-i.shape[-1]), 0)(i) for i in pitch]
    batch_speakers = torch.Tensor([i[1] for i in data]).long()


    return {"x":torch.stack(padded_mels), "p": torch.stack(padded_pitch), "spk_id":batch_speakers}