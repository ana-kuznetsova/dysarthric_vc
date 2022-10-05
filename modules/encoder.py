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
                feature_extractor,
                feat_extractor_dim,
                hidden_dim, batch_size,
                num_classes,
                mi=False

    ):
        super(GeneralEncoder, self).__init__()

        self.batch_size = batch_size
        self.feature_extractor = feature_extractor
        self.speaker_encoder = MLP(feat_extractor_dim, 
                                   hidden_dim, hidden_dim)
        self.speaker_cls = nn.Linear(hidden_dim, num_classes)
        '''
        self.pitch_predictor = nn.GRU(input_size=1, hidden_size=hidden_dim, 
                                      num_layers=1)
                                      '''
        #Predicts everything else other than pitch or speaker ID
        self.attr_predictor = MLP(feat_extractor_dim, 
                                   hidden_dim, hidden_dim)
        self.reconstructor = nn.Linear(hidden_dim*2, feat_extractor_dim)
        self.activation_1 = nn.LeakyReLU()
        self.activation_2 = nn.Softmax(dim=1)
        self.activation_3 = nn.LeakyReLU()
        self.mi = mi
        


    def forward(self, x):
        feats = self.feature_extractor(x)
        #All about speaker ID
        spk_embed = self.speaker_encoder(feats)
        spk_embed = self.activation_1(spk_embed)
        spk_cls_out = self.activation_2(self.speaker_cls(spk_embed))

        #All about pitch
        '''
        pitch_inp = torch.cat((feats, p), dim=1)
        pitch_inp = torch.transpose(pitch_inp, 0, 1).unsqueeze(2)
        out, h_n = self.pitch_predictor(pitch_inp)
        pitch_embed = h_n.squeeze(0)
        '''

        #Other attrs
        attr_embed = self.activation_3(self.attr_predictor(feats))
        #concat_embed = torch.cat((spk_embed, pitch_embed, attr_embed), dim=1)
        concat_embed = torch.cat((spk_embed, attr_embed), dim=1)
        proj = self.reconstructor(concat_embed)
        return {"feats":feats, "proj":proj, "spk_cls":spk_cls_out, "spk_emb":spk_embed, "attr_emb":attr_embed}