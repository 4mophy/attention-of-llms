# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/2 17:47
# @File: EncoderLayer.py
# @Author: 10993
# @Description: 
"""
from torch import nn

from FeedForward import FeedForward
from MultiHeadedAttention import MultiHeadedAttention


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadedAttention(d_model, num_heads, dropout)

        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None):
        x = src
        x = x + self.attn(x, x, x, mask=src_mask)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x
