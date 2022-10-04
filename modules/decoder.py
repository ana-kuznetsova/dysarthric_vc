#from speechbrain.pretrained import Tacotron2
from speechbrain.lobes.models.Tacotron2 import Tacotron2
import torch

class Tacotron2Conditional(Tacotron2):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, condition_embed, alignments_dim=None):
        #'Alingnments dim needed for data parallel'
        inputs, input_lengths, targets, max_len, output_lengths = inputs
        input_lengths, output_lengths = input_lengths.data, output_lengths.data

        embedded_inputs = self.embedding(inputs).transpose(1, 2)

        encoder_outputs = self.encoder(embedded_inputs, input_lengths)

        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths
        )

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            output_lengths,
            alignments_dim,
        )


#ckpt =  "/home/anakuzne/exp/tacotron2/fine_tune/1234/save/CKPT+2022-09-28+14-13-06+00/model.ckpt"
#model = Tacotron2Conditional() 
#print(model)
'''
taco_ckpt = '/home/anakuzne/exp/tacotron2/fine_tune/1234'
taco = Tacotron2.from_hparams(source=taco_ckpt, 
                                   savedir="/home/anakuzne/exp/tacotron2/fine_tune/1234/save/CKPT+2022-09-28+14-13-06+00/",
                                   run_opts={"device":"cuda"})
print(taco.text_cleaners)
'''