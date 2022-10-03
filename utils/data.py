import torch.utils.data as data
import torch
import torch.nn as nn
import os
import numpy as np
import librosa
import torchaudio
from torchaudio import transforms
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


def filter_speakers(config):
    test_spk = list(config.data.test_partition)
    ignore = config.data.ignore_speakers
    all_wav_paths = collect_fnames(config.data.dataset_path)
    if ignore[0]:
        all_wav_paths = [f for f in all_wav_paths for i in ignore if i not in f]
    test_files = []
    train_files = []

    #Collect train, test file lists, unique speakers
    if config.data.dataset=='VCTK':
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
    elif config.data.dataset=='DysarthricSim':
        with open(config.data.meta_path, 'r') as fo:
            meta = fo.readlines()
        
        file2spk_map = {}
        true_files = os.listdir(config.data.dataset_path)
        for line in meta:
            f = line.split('|')[0]+'.wav'
            if f in true_files:
                f = os.path.join(config.data.dataset_path, f)
                spk = line.split("|")[1]
                file2spk_map[f] = spk 
        
        for k in file2spk_map:
            path = os.path.join(config.data.dataset_path, k)
            if file2spk_map[k] in config.data.test_partition:
                test_files.append(path)
            else:
                train_files.append(path)
        unique_speakers = list(set(file2spk_map.values()))
    
    if ignore[0]:
        for i in ignore:
            unique_speakers.remove(i)
            if os.path.exists('log.txt'):
                unique_speakers.remove('log.txt')

    spk2id_map = {}

    for i in range(len(unique_speakers)):
        spk2id_map[unique_speakers[i]] = i
    
    print(f"> Found {len(spk2id_map.keys())} unique speakers, skipping {ignore}...")

    train_spk_ids = []
    test_spk_ids = []
    
    if config.data.dataset=='VCTK':
        for i in train_files:
            spk_id = i.split('/')[-1].split('_')[0].replace('.wav', '')
            train_spk_ids.append(spk2id_map[spk_id])

        for i in test_files:
            spk_id = i.split('/')[-1].split('_')[0].replace('.wav', '')
            test_spk_ids.append(spk2id_map[spk_id])
    elif config.data.dataset=='DsyarthricSim':
        for i in train_files:
            wav = i.split('/')[0]
            spk_id = file2spk_map[wav]
            train_spk_ids.append(spk2id_map[spk_id])
        for i in test_files:
            wav = i.split('/')[0]
            spk_id = file2spk_map[wav]
            test_spk_ids.append(spk2id_map[spk_id])

    #print(f"DEBUG {train_files[:5]}")
    #print(f"DEBUG {spk2id_map}")
    if config.data.dataset=='VCTK':
        return train_files, train_spk_ids,  test_files, test_spk_ids, spk2id_map, None
    return train_files, train_spk_ids,  test_files, test_spk_ids, spk2id_map, file2spk_map


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
        
        train_files, train_spk_ids, test_files, test_spk_ids, spk2id_map, _ = filter_speakers(config)
        self.train_files = train_files
        self.train_spk_ids = train_spk_ids
        self.test_files = test_files
        self.test_spk_ids = test_spk_ids
        self.text_train = []
        self.text_test = []

        with open(config.data.text_path, 'r') as fo:
            text_dict = json.load(fo)

        for f in train_files:
            fname = f.split('/')[-1].replace('.wav', '')
            sent = text_dict[fname]["label"]
            self.text_train.append(sent)

        for f in test_files:
            fname = f.split('/')[-1].replace('.wav', '')
            sent = text_dict[fname]["label"]
            self.text_test.append(sent)

    def __len__(self):
        if self.mode=='train':
            return len(self.train_files)
        return len(self.test_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.mode=='train':
            return (self.train_files[idx], self.train_spk_ids[idx], self.text_train[idx])
        return (self.test_files[idx], self.test_spk_ids[idx], self.text_test[idx])


class VCTKAngleProtoData(data.Dataset):
    def __init__(self, config, mode='train'):
        self.mode = mode
        self.data_path = config.data.dataset_path
        
        train_files, train_spk_ids, test_files, test_spk_ids, spk2id_map, file2spk_map = filter_speakers(config)

        assert config.model.num_speakers*config.model.num_utter == config.trainer.batch_size
        #map speakers to files 

        spk2file_map = {s:[] for s in spk2id_map}

        if config.data.dataset=='VCTK':
            for s in spk2id_map:
                for f in train_files:
                    if s in f:
                        spk2file_map[s].append(f)
        elif config.data.dataset=='DysarthricSim':
            for f in file2spk_map:
                spk = file2spk_map[f]
                spk2file_map[spk].append(f)
        
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

        if config.data.dataset=='DysarthricSim':
            #Rearrange spk2id_map based on discarded speakers
            train_speakers = list(set([file2spk_map[i] for i in train_arranged]))
            spk2id_map = {s:i for i, s in enumerate(train_speakers)}
    
        for i in train_arranged:
            if config.data.dataset=='VCTK':
                spk_id = i.split('/')[-1].split('_')[0]
            elif config.data.dataset=='DysarthricSim':
                spk_id = file2spk_map[i]
            train_spk_ids_arranged.append(spk2id_map[spk_id])
        print(f"> Number of unique speakers in train {len(set(train_spk_ids_arranged))}")

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


def collate_spk_enc(data):
    '''
    For batch
    '''
    sr=16000
    mel_specs = []
    transform = transforms.MelSpectrogram(sr, n_mels=80)

    for fname in data:
        #wav, fs = librosa.load(fname[0], sr=16000)
        wav, _ = torchaudio.load(fname[0])
        #spec = librosa.feature.melspectrogram(y=wav, sr=fs, n_mels=80)
        spec = transform(wav)
        #mel_specs.append(torch.Tensor(spec))
        mel_specs.append(spec)
    
    maxlen_mel = max([i.shape[-1] for i in mel_specs])
    padded_mels = [nn.ZeroPad2d(padding=(0, maxlen_mel - i.shape[-1], 0, 0))(i) for i in mel_specs]
    
    batch_speakers = torch.Tensor([i[1] for i in data]).long()


    return {"x":torch.stack(padded_mels), "spk_id":batch_speakers}


def pad_noise(speech, noise):
    '''
    Cuts noise vector if speech vec is shorter
    Adds noise if speech vector is longer
    '''
    noise_len = noise.shape[1]
    speech_len = speech.shape[1]

    if speech_len > noise_len:
        repeat = (speech_len//noise_len) +1
        noise = torch.tile(noise, (1, repeat))
        diff = speech_len - noise.shape[1]
        noise = noise[:, :noise.shape[1]+diff]          
            
    elif speech_len < noise_len:
        noise = noise[:,:speech_len]
    return noise

def mix_signals(speech, noise, desired_snr):    
    #calculate energies
    E_speech = torch.sum(speech**2)
    E_noise = torch.sum(noise**2)
    
    #calculate b coeff
    b = torch.sqrt((E_speech/((desired_snr/10)**10))/E_noise)    
    return speech + b*noise



def augment(fname):
    musan_files = []
    path = '/data/common/musan'

    for root, d, files in os.walk(path):
        for f in files:
            fpath = os.path.join(root, f)
            if '.wav' in fpath:
                musan_files.append(fpath)

    #orig_wav, fs = librosa.load(fname[0], sr=16000)
    orig_wav, fs = torchaudio.load(fname[0])

    return_values = []
    for i in range(2):
        SNR = random.randrange(13, 20)
        #mix_sig, fs = librosa.load(random.choice(musan_files), sr=16000)
        mix_sig, fs = torchaudio.load(random.choice(musan_files))
        mix_sig = pad_noise(orig_wav, mix_sig)
        mix_wav = mix_signals(orig_wav, mix_sig, SNR)
        return_values.append(mix_wav)
    
    return [orig_wav] + return_values


def collate_spk_enc_augment(data):
    '''
    For batch
    '''

    sr=16000
    mel_specs = []
    transform = transforms.MelSpectrogram(sr, n_mels=80)

    for fname in data:
        mixed_wavs = augment(fname)
        for wav in mixed_wavs:
            spec = transform(wav)
            mel_specs.append(spec)
    
    maxlen_mel = max([i.shape[-1] for i in mel_specs])
    padded_mels = [nn.ZeroPad2d(padding=(0, maxlen_mel - i.shape[-1], 0, 0))(i) for i in mel_specs]
    
    batch_speakers = []
    for i in data:
        i = [i[1]]*3
        batch_speakers.extend(i)
    batch_speakers = torch.Tensor(batch_speakers).long()
    return {"x":torch.stack(padded_mels), "spk_id":batch_speakers}