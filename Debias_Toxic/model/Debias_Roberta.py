from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F

from transformers.modeling_bert import BertEmbeddings, BertModel, BertPreTrainedModel
from transformers.configuration_roberta import RobertaConfig
from transformers.file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}

class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size,
                                                padding_idx=self.padding_idx)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(self.padding_idx+1, seq_length+self.padding_idx+1, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        # print(self.word_embeddings(torch.tensor(50264)))  # <mask> 的初始权重
        return super(RobertaEmbeddings, self).forward(input_ids,
                                                      token_type_ids=token_type_ids,
                                                      position_ids=position_ids,
                                                      inputs_embeds=inputs_embeds)

class RobertaModel(BertModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value


# 分类器输出维度为 2，损失函数 CrossEntropyLoss

class RobertaForDebiasSequenceClassification(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForDebiasSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config)
        
        self.bias_lin = torch.nn.Linear(768, 1)  # bias 影响因子
        # self.device = torch.device('cuda:0')
    
    def debias_loss(self, hidden, logits, bias, labels, penalty=0.01):

        logits = logits.float()  # In case we were in fp16 mode  [batch_size, len, dim]
        hidden = hidden[:, 0, :] # Take <s>, the same as CLS token  [batch_size, dim]
        logits = F.log_softmax(logits, 1)  # [batch_size, dim]
        factor = self.bias_lin.forward(hidden)  # factor 是由 hidden 过线性层得到的
        factor = factor.float()
        factor = F.softplus(factor)  # softplus: 激活函数
        bias = bias * factor  
        bias_lp = F.log_softmax(bias, 1)
        entropy = -(torch.exp(bias_lp) * bias_lp).sum(1).mean(0)
        loss = F.cross_entropy(logits + bias, labels) + penalty*entropy
        return loss
        #return loss, logits+bias

    def forward(self, setting, if_bias=False, **kwargs):

        outputs = self.roberta(input_ids=kwargs['text_idx'].to(setting.device), 
                               attention_mask=kwargs['text_mask'].to(setting.device))
        # outputs = self.roberta(input_ids=kwargs['emb_idx'].to(self.device),
        #                        attention_mask=kwargs['emb_mask'].to(setting.device))
        # outputs = self.roberta(input_ids=kwargs['text_emb_idx'].to(self.device),
        #                        attention_mask=kwargs['text_emb_mask'].to(setting.device))

        sequence_output = outputs[0]
        # print(sequence_output.shape)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]   # logits + (hidden_states, attentions)
        
        labels = kwargs["label"].to(setting.device)

        if not if_bias:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
        else:
            bias = kwargs["bias"].to(setting.device) 
            loss = self.debias_loss(sequence_output, logits, bias, labels)
        outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x