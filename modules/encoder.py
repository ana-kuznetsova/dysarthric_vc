import torch
from torch import nn

from TTS.encoder.models.resnet import ResNetSpeakerEncoder


class DysarthEncoder(nn.Module):
    def __init__(self, config):
        super(DysarthEncoder, self).__init__()
        self.SpeakerEncoder = ResNetSpeakerEncoder(input_dim=config.data.feature_dim)

    def forward(self, x):
        x = self.SpeakerEncoder(x)
        return x