import torch
from torch import nn
from  torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self,
                inp_feature_dim, 
                hidden_dim,
                out_dim
    ):
        super(MLP, self).__init__()
        self.mlp_fc1 = nn.Linear(inp_feature_dim, hidden_dim)
        self.mlp_fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.mlp_fc1(x)
        x = F.leaky_relu(x)
        x = self.mlp_fc2(x)
        x = F.leaky_relu(x)
        return x


class GeneralEncoder(nn.Module):
    '''
    General encoder does not model dysarthric featrures
    It only models speaker ID embedding and Pitch predictor
    '''

    def __init__(self,
                inp_feature_dim,
                feature_extractor,
                feat_extractor_dim,
                hidden_dim, batch_size

    ):
        super(GeneralEncoder, self).__init__()

        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.speaker_encoder = MLP(feat_extractor_dim, 
                                   hidden_dim, hidden_dim)
        self.pitch_predictor = nn.GRU(input_size=1, hidden_size=hidden_dim, 
                                      num_layers=1)
        self.reconstructor = nn.Linear(hidden_dim*2, feat_extractor_dim)
        


    def forward(self, x, p):
        feats = self.feature_extractor(x)
        spk_embed = self.speaker_encoder(feats)
        spk_embed = F.leaky_relu(spk_embed)
        pitch_inp = torch.cat((feats, p), dim=1)
        pitch_inp = torch.transpose(pitch_inp, 0, 1).unsqueeze(2)
        out, h_n = self.pitch_predictor(pitch_inp)
        pitch_embed = h_n.squeeze(0)
        concat_embed = torch.cat((spk_embed, pitch_embed), dim=1)
        proj = self.reconstructor(concat_embed)
        return feats, proj