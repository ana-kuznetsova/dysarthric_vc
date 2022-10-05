import torch
from torch import nn
from  torch.nn import functional as F
from utils.utils import shuffle_tensor
from speechbrain.lobes.models.Tacotron2 import Loss as TacoLoss

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
    Taken from the original implementation of CLUB at
    https://github.com/Linear95/CLUB/

        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size//2, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)



class LossGeneral(nn.Module):
    '''
    This class computes the loss over basic setup of the encoder
    The loss is the weighted sum over embedding reconstru
    '''
    def __init__(self, alpha1=1, alpha2=1, guided_attention_scheduler=None):
        super(LossGeneral, self).__init__()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.rc_loss = nn.L1Loss()
        self.spk_ce_loss = nn.CrossEntropyLoss()
        '''
            gate_loss_weight: 1.0
            guided_attention_weight: 50.0
            guided_attention_sigma: 0.2
            guided_attention_scheduler: *id001
            guided_attention_hard_stop: 50

            guided_attention_sigma=None,
            gate_loss_weight=1.0,
            guided_attention_weight=1.0,
            guided_attention_scheduler=None,
            guided_attention_hard_stop=None,

        '''
        self.taco_loss = TacoLoss(guided_attention_sigma=0.2,
                                  gate_loss_weight=1.0
                                  guided_attention_weight=50.0,
                                  guided_attention_scheduler=guided_attention_scheduler,
                                  guided_attention_hard_stop=10.0
                                  )

    def forward(self, x, cls_target, outs):
        mels_pred = outs['mel_outputs']
        mels_pred_postnet = outs['mel_outputs_postnet']
        cls_out = outs['spk_cls']
       
        loss_1 = self.alpha1*self.rc_loss(x, mels_pred)
        loss_2 = self.alpha2*self.spk_ce_loss(cls_out, cls_target)
        taco_loss = self.taco_loss(mels_pred, mels_pred_postnet)
        total_loss = loss_1 + loss_2 + taco_loss
        return total_loss, loss_1, loss_2, taco_loss