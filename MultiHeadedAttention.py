# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/2 12:52
# @File: MultiHeadedAttention.py
# @Author: 10993
# @Description: 多头注意力
"""
import torch
from torch import nn

from SelfAttention import SelfAttention


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 每个头的维度是总维度除以头数
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        self.attentions = nn.ModuleList([
            SelfAttention(d_model, self.head_dim)
            for _ in range(self.num_heads)
        ])

        self.output = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 收集所有头的输出
        head_outputs = []
        for layer in self.attentions:
            head_outputs.append(layer(q, k, v, mask))

        # 在最后一个维度上连接所有头的输出
        x = torch.cat(head_outputs, dim=-1)
        x = self.output(x)
        return x
