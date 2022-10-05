#from speechbrain.pretrained import Tacotron2
from speechbrain.lobes.models.Tacotron2 import Tacotron2
from speechbrain.lobes.models.Tacotron2 import Loss as TacoLoss
import torch
from torch import nn

class Interface(nn.Module):
    def __init__(self):
        super().__init__()
        self.interface = nn.Linear(768, 512)

    def forward(self, x):
        return self.interface(x)


class Tacotron2Conditional(Tacotron2):
    def __init__(self):
        super().__init__()

    def forward(self, text, targets, condition_embed, interface, device='cuda:0', alignments_dim=None):
        #'Alingnments dim needed for data parallel'

        inputs = text['text_sequences'].data
        max_len = max([i.shape[0] for i in inputs])
        input_lengths = text['text_sequences'].lengths
        input_lengths = torch.tensor([i*max_len for i in input_lengths])

        output_lengths = targets['mel_specs'].lengths
        targets = targets['mel_specs'].data
        
        targets = torch.transpose(targets, 1, 2)
        
        max_len = max([i.shape[1] for i in targets])
        output_lengths = torch.tensor([i*max_len for i in output_lengths])

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)
        num_repeats = encoder_outputs.shape[1]
        
        condition_embed = condition_embed.unsqueeze(1)
        condition_embed = torch.tile(condition_embed, (1, num_repeats, 1))
        encoder_outputs = torch.cat((encoder_outputs, condition_embed), dim=2)
        encoder_outputs = interface(encoder_outputs)

        input_lengths = input_lengths.to(device)
        output_lengths = output_lengths.to(device)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
   
        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths,
            alignments_dim,
        ), targets, input_lengths, output_lengths, mel_outputs_postnet, mel_outputs

'''
ckpt =  "/home/anakuzne/exp/tacotron2/fine_tune/1234/save/CKPT+2022-09-28+14-13-06+00/model.ckpt"
model = Tacotron2Conditional() 
for n, param in model.named_parameters():
    print(n)


taco_ckpt = '/home/anakuzne/exp/tacotron2/fine_tune/1234'
taco = Tacotron2.from_hparams(source=taco_ckpt, 
                                   savedir="/home/anakuzne/exp/tacotron2/fine_tune/1234/save/CKPT+2022-09-28+14-13-06+00/",
                                   run_opts={"device":"cuda"})
print(taco.text_cleaners)
'''