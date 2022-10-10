from modules.encoder import GeneralEncoder
from modules.decoder import Tacotron2Conditional
from torch import nn
import torch


class JointVC(nn.Module):
    '''
    E2E encoder decoder model for spkeaker attr disentanglement
    Decoder: Tacotron2 should be initialized with pretrained weights
    '''

    def __init__(self,
                encoder,
                decoder,
    ):
        super(JointVC, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, text, target, interface):
        outputs = self.encoder(x)
        #{"feats":feats, "proj":proj, "spk_cls":spk_cls_out, "spk_emb":spk_embed, "attr_emb":attr_embed}
        spk_embed = outputs["spk_emb"]
        attr_embed = outputs["attr_emb"]
        cond_embed = torch.cat((spk_embed, attr_embed), dim=1)

        decoder_outputs, targets, input_lengths, target_lengths, mels_pred_postnet, mel_outputs = self.decoder(text, target, cond_embed, interface)
        assert x.shape[2]==mels_pred_postnet.shape[2], f"input x of shape {x.shape} does not match mels_postnet {mels_pred_postnet.shape}"
        outputs["decoder_outputs"] = decoder_outputs
        outputs["input_lengths"] =  input_lengths
        outputs["target_lengths"] = target_lengths
        outputs["targets"] = target
        outputs["mels_pred_postnet"] = mels_pred_postnet
        outputs["mels_pred"] = mel_outputs
        return outputs