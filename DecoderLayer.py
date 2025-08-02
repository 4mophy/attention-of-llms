# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/2 03:17
# @File: DecoderLayer.py
# @Author: 10993
# @Description: 解码器层
"""
from torch import nn

from FeedForward import FeedForward
from MultiHeadedAttention import MultiHeadedAttention


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()

        # 多头注意力机制
        self.masked_attn = MultiHeadedAttention(d_model, num_heads, dropout)

        # 归一化
        self.masked_attn_norm = nn.LayerNorm(d_model)

        self.attn = MultiHeadedAttention(d_model, num_heads, dropout)

        self.attn_norm = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, enc, tgt_mask=None, enc_mask=None):
        x = tgt
        x = x + self.masked_attn(x, x, x, mask=tgt_mask)
        x = self.masked_attn_norm(x)
        x = self.attn(x, enc, enc, mask=enc_mask)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        x = self.ffn_norm(x)
        return x
