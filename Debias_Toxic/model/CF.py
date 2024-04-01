import os
import torch
import torch.nn as nn
import tqdm
from model.layers import *
from sklearn.metrics import *
from transformers import BertModel, RobertaModel, AutoModel
# from utils.utils import data2gpu, Averager, metrics, Recorder
# from utils.dataloader import get_dataloader

class BERT_CFModel(nn.Module):
    def __init__(self, config):
        # emb_dim, mlp_dims, dropout
        super(BERT_CFModel, self).__init__()
        self.device = config.device
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.mode = config.mode

        self.mlp_text = MLP(config.vocab_dim, config.mlp_dim, config.dropout)
        self.mlp_emb = MLP(config.vocab_dim, config.mlp_dim, config.dropout)
        self.mlp_text_emb = MLP(config.vocab_dim, config.mlp_dim, config.dropout)
        self.attention = MaskAttention(config.vocab_dim)  # 自掩码注意力
        self.att = Attention()  # 点乘注意力

        self.mixup = Mixup(config)
    
    def forward(self, **kwargs):
        inputs = kwargs['text_idx'].to(self.device)
        masks = kwargs['text_mask'].to(self.device)
        roberta_feature = self.roberta(inputs)[0]
        feature, _ = self.attention(roberta_feature, masks)
        s_pred = self.mlp_text(feature).squeeze(1)  

        entity = kwargs['emb_idx'].to(self.device)
        masks = kwargs['emb_mask'].to(self.device)
        entity_feature = self.roberta(entity)[0]
        # w_pred = self.entity_net(entity_feature).squeeze(1)
        entity_feature, _ = self.attention(entity_feature, masks)
        entity_feature = grad_mul_const(entity_feature, 0.0)
        w_pred = self.mlp_emb(entity_feature).squeeze(1)  
  
        ensemble_feature = self.att(query=feature, key=entity_feature, value=feature)
        # print(ensemble_feature.shape)
        logits_k = self.mlp_text_emb(ensemble_feature).squeeze(1)  

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
        self.constant = nn.Parameter(torch.tensor(0.0))  # constant 指代反事实下各个分支模型的先验概率，可以设为定值，也可以训练优化
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