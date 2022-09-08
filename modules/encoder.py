import torch
from torch import nn
from  torch.nn import functional as F

from TTS.encoder.models.resnet import ResNetSpeakerEncoder


class DysarthEncoder(nn.Module):
    def __init__(self,
                inp_feature_dim,
                spk_encoder_dim,
                spk_id_dim,
                pitch_dim,
                dysarth_dim,
                proj_dim
    ):
        super(DysarthEncoder, self).__init__()
        self.spk_encoder = ResNetSpeakerEncoder(input_dim=inp_feature_dim)
        self.spk_info_emb = nn.Linear(spk_encoder_dim, spk_id_dim)
        self.f0_embed = nn.Linear(spk_encoder_dim, pitch_dim)
        self.dysarth_embed = nn.Linear(spk_encoder_dim, dysarth_dim)
        self.proj = nn.Linear(spk_id_dim + pitch_dim + dysarth_dim, inp_feature_dim)

    def forward(self, inp):
        x = self.SpeakerEncoder(inp)
        spk_info = F.tanh(self.spk_info_emb(x))
        f0 = F.leaky_relu(self.f0_embed(x))
        dysarth_info = F.tanh(self.dysarth_embed)
        concat = torch.cat((spk_info, f0, dysarth_info))
        print(f"dysarth {dysarth_info.shape}")

        return x 