import os
import torch
import torch.nn as nn
import tqdm
from model.layers import *
from sklearn.metrics import *
from transformers import BertModel, RobertaModel
# from utils.utils import data2gpu, Averager, metrics, Recorder
# from utils.dataloader import get_dataloader

class BERT_CFModel(nn.Module):
    def __init__(self, config):
        # emb_dim, mlp_dims, dropout
        super(BERT_CFModel, self).__init__()
        self.device = config.device
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # self.embedding = self.roberta.embeddings
        self.mode = config.mode
        
        # for name, param in self.bert.named_parameters():
        #     if name.startswith("encoder.layer.11"): \
        #             #or name.startswith('encoder.layer.10') \
        #             #or name.startswith('encoder.layer.9'): \
        #             # or name.startswith('encoder.layer.8') \
        #             # or name.startswith('encoder.layer.7') \
        #             # or name.startswith('encoder.layer.6')\
        #             # or name.startswith('encoder.layer.5') \
        #             # or name.startswith('encoder.layer.4')\
        #             # or name.startswith('encoder.layer.3'):
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False

        self.mlp = MLP(config.vocab_dim, config.mlp_dim, config.dropout)
        self.attention = MaskAttention(config.vocab_dim)
        
        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        self.entity_convs = cnn_extractor(feature_kernel, config.vocab_dim)
        mlp_input_shape = sum([feature_kernel[kernel] for kernel in feature_kernel])
        self.entity_mlp = MLP(mlp_input_shape, config.mlp_dim, config.dropout)
        self.entity_net = nn.Sequential(self.entity_convs, self.entity_mlp)

        self.mixup = Mixup(config)
    
    def forward(self, **kwargs):
        inputs = kwargs['text_idx'].to(self.device)
        masks = kwargs['text_mask'].to(self.device)
        roberta_feature = self.roberta(inputs)[0]
        feature, _ = self.attention(roberta_feature, masks)
        s_pred = self.mlp(feature).squeeze(1)  

        entity = kwargs['emb_idx'].to(self.device)
        masks = kwargs['emb_mask'].to(self.device)
        entity_feature = self.roberta(entity)[0]
        entity_feature = grad_mul_const(entity_feature, 0.0)
        w_pred = self.entity_net(entity_feature).squeeze(1)
   
        inputs = kwargs['text_emb_idx'].to(self.device)
        masks = kwargs['text_emb_mask'].to(self.device)
        feature = self.roberta(inputs)[0]
        feature = grad_mul_const(feature, 0.0)
        feature, _ = self.attention(feature, masks)
        logits_k = self.mlp(feature).squeeze(1)  

        # both k and w are the facts
        z_swk = self.mixup.fusion(logits_k, w_pred, s_pred, w_fact=True, k_fact=True, s_fact=True) # Z_{w, k} 

        out = {}
        # z_k, z_w, _
        z_w = self.mixup.fusion(logits_k, w_pred, s_pred, w_fact=True, k_fact=False, s_fact=False) # Z_{w, k*} 
        logits_cf = z_swk - z_w  # TIE = TE - NDE = Z_{w, s, k} - Z_{w, s*, k*}

        out['logits_all'] = z_swk # for optimization
        out['logits_k']  = logits_k # predictions of the original sentence-word branch, i.e., NIE
        out['logits_cf'] = logits_cf # predictions of CF, i.e., TIE = TE - NDE
        out['logits_w'] = w_pred # for optimization
        out['logits_s'] = s_pred # for optimization

        return out


class Mixup():

    def __init__(self, config):
        self.fusion_mode = config.fusion_mode
        self.constant = nn.Parameter(torch.tensor(0.0))  
        self.eps = config.eps

    def fusion(self, z_k, z_w, z_s, w_fact=True, k_fact=True, s_fact=True):
        """
        z_k: the predictions of the sentence-word branch
        z_w: the predictions of the word-only branch
        """
        z_k, z_w, z_s = self.transform(z_k, z_w, z_s, w_fact, k_fact, s_fact)

        if self.fusion_mode == 'rubi':
            z = z_k * torch.sigmoid(z_w)

        elif self.fusion_mode == 'hm':
            z = torch.sigmoid(z_k) * torch.sigmoid(z_w) * torch.sigmoid(z_s)
            z = torch.log(z + self.eps) - torch.log1p(z)  # log1p(z) = log(1+z)

        elif self.fusion_mode == 'sum':
            z = z_k + z_w + z_s
            z = torch.log(torch.sigmoid(z) + self.eps)

        return z
    
    def transform(self, z_k, z_w, z_s, w_fact=False, k_fact=False, s_fact=True):

        if not k_fact:
            z_k = self.constant * torch.ones_like(z_k)

        if not w_fact:
            z_w = self.constant * torch.ones_like(z_w)

        if not s_fact:
            z_s = self.constant * torch.ones_like(z_s)

        return z_k, z_w, z_s

# if __name__ == "__main__":
#     config = Config_base("roberta-base", "toxic_lang_data_pub")
#     BERT_ENDEFModel(config)