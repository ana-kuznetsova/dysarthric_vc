from modules.encoder import GeneralEncoder
from modules.decoder import Tacotron2Conditional




class EncoderDecoder(nn.Module):
    '''
    E2E encoder decoder model for spkeaker attr disentanglement
    Decoder: Tacotron2 should be initialized with pretrained weights
    '''

    def __init__(self,
                feature_extractor,
                decoder,
                feat_extractor_dim,
                hidden_dim, batch_size
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder = GeneralEncoder(feat_extractor, 
                                      feat_extractor_dim, 
                                      hidden_dim, batch_size, num_classes)
        self.decoder = decoder

    def forward(self, x, t):
        outputs = self.encoder(x)
        #{"feats":feats, "proj":proj, "spk_cls":spk_cls_out, "spk_emb":spk_embed, "attr_emb":attr_embed}
        spk_embed = outputs["epk_emb"]
        attr_embed = outputs["attr_emb"]
        cond_embed = torch.cat((spk_embed, attr_embed), dim=1)
        mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.decoder(t, cond_embed)

        outputs["mel_outputs"] = mel_outputs
        outputs["mel_outputs_postned"] = mel_outputs_postnet
        return outputs