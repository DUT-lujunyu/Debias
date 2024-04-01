import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math

class GradMulConst(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)

# class MLP(torch.nn.Module):

#     def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
#         super().__init__()
#         layers = list()
#         for embed_dim in embed_dims:
#             layers.append(torch.nn.Linear(input_dim, embed_dim))
#             layers.append(torch.nn.ReLU())
#             layers.append(torch.nn.Dropout(p=dropout))
#             input_dim = embed_dim
#         if output_layer:
#             layers.append(torch.nn.Linear(input_dim, 2))
#         self.mlp = torch.nn.Sequential(*layers)

#     def forward(self, x):
#         """
#         :param x: Float tensor of size ``(batch_size, embed_dim)``
#         """
#         return self.mlp(x)

# 与 RobertaClassificationHead 统一的写法
class MLP(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super(MLP, self).__init__()
        self.dense = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class cnn_extractor(nn.Module):
    def __init__(self, feature_kernel, input_size):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()])

    def forward(self, input_data):
        share_input_data = input_data.permute(0, 2, 1)
        feature = [conv(share_input_data) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = feature.view([-1, feature.shape[1]])
        return feature


class MaskAttention(torch.nn.Module):
    """
    Compute attention layer
    """
    def __init__(self, input_shape):
        super(MaskAttention, self).__init__()
        self.attention_layer = torch.nn.Linear(input_shape, 1)

    def forward(self, inputs, mask=None):
        # print("inputs: ", inputs.shape)     #(128, 170, 768)
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        # print("scores: ", scores.shape)     #(128, 170)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        scores = torch.softmax(scores, dim=-1).unsqueeze(1)
        # print("scores: ", scores.shape)     #(128, 1, 170)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        # print("outputs: ", outputs.shape)   #(128, 768)

        return outputs, scores


class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value) #, p_attn