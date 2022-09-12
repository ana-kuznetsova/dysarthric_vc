import torch.utils.data as data
import torch
import torch.nn as nn
import os
import numpy as np
import librosa
import parselmouth
import json
from attrdict import AttrDict
import random



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
    snd = parselmouth.Sound(path[0]).resample(16000)
    pitch = snd.to_pitch()
    pitch_arr = []
    for i in range(pitch.n_frames):
        pitch_arr.append(pitch.get_value_in_frame(i))
    return np.nan_to_num(np.array(pitch_arr))


def filter_speakers(configs):
    test_spk = list(configs.data.test_partition)
    ignore = configs.data.ignore_speakers
    all_wav_paths = collect_fnames(configs.data.dataset_path)
    all_wav_paths = [f for f in all_wav_paths for i in ignore if i not in f]
    test_files = []
    train_files = []

    for f in all_wav_paths:
        flag = False
        for t in test_spk:
            if t in f:
                flag=True
                test_files.append(f)
                break    
        if flag==False:
            train_files.append(f)
                

    unique_speakers = os.listdir(configs.data.dataset_path)
    for i in ignore:
        unique_speakers.remove(i)
        unique_speakers.remove('log.txt')

    spk2id_map = {}

    for i in range(len(unique_speakers)):
        spk2id_map[unique_speakers[i]] = i
    
    print(f"> Found {len(spk2id_map.keys())} unique speakers, skipping {ignore}...")

    train_spk_ids = []
    test_spk_ids = []
    
    for i in train_files:
        spk_id = i.split('/')[-1].split('_')[0]
        train_spk_ids.append(spk2id_map[spk_id])

    for i in test_files:
        spk_id = i.split('/')[-1].split('_')[0]
        test_spk_ids.append(spk2id_map[spk_id])

    return train_files, train_spk_ids,  test_files, test_spk_ids, spk2id_map

class LibriTTSData(data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.data_path = config.data.dataset_path
        self.train_path = os.path.join(self.data_path, config.data.train_partition)
        self.test_path = os.path.join(self.data_path, config.data.test_partition)
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
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.data_path = config.data.dataset_path
        
        train_files, train_spk_ids, test_files, test_spk_ids, spk2id_map = filter_speakers(configs)
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


class VCTKAngleProtoData(data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.data_path = config.data.dataset_path
        
        train_files, train_spk_ids, test_files, test_spk_ids, spk2id_map = filter_speakers(config)

        assert config.model.num_speakers*config.model.num_utter == config.trainer.batch_size
        #map speakers to files 

        spk2file_map = {s:[] for s in spk2id_map}

        for s in spk2id_map:
            for f in train_files:
                if s in f:
                    spk2file_map[s].append(f)
        
        speakers = list(spk2file_map.keys())

        for k in speakers:
            if k in list(config.data.test_partition):
                del spk2file_map[k]

        speakers = list(spk2file_map.keys())
        train_arranged = []

        while len(speakers)>0:
            #Check if all speakers have enough utters
            for s in speakers:
                if len(spk2file_map[s]) < config.model.num_utter:
                    speakers.remove(s)

            if len(speakers) < config.model.num_speakers:
                break

            for i in range(config.model.num_speakers):
                spk = random.choice(speakers)
                utters = []
                for i in range(config.model.num_utter):
                    if len(spk2file_map[spk])==0:
                        break
                    utters.append(spk2file_map[spk][0])
                    spk2file_map[spk].pop(0)
                train_arranged.extend(utters)

        print(f"> Number of train utterances: {len(train_arranged)}")
        train_spk_ids_arranged = []
        for i in train_arranged:
            spk_id = i.split('/')[-1].split('_')[0]
            train_spk_ids_arranged.append(spk2id_map[spk_id])

        self.train_files = train_arranged
        self.train_spk_ids = train_spk_ids_arranged
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