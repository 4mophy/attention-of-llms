# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/2 12:59
# @File: SelfAttention.py
# @Author: 10993
# @Description: 
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class SelfAttention(nn.Module):
    def __init__(self, d_model, head_dim, dropout=0.1):
        super().__init__()
        self.query = nn.Linear(d_model, head_dim)
        self.key = nn.Linear(d_model, head_dim)
        self.value = nn.Linear(d_model, head_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # 线性层
        query = self.query(q)
        key = self.key(k)
        value = self.value(v)

        # 矩阵乘法
        dim_k = key.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(dim_k)

        # 掩码
        if mask is not None:
            # 调整mask的维度以匹配scores
            if mask.dim() != scores.dim():
                # 扩展mask的维度
                for _ in range(scores.dim() - mask.dim()):
                    mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -float('inf'))

        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        outputs = torch.matmul(weights, value)
        return outputs
