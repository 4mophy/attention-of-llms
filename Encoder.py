# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/2 02:31
# @File: Encoder.py
# @Author: 10993
# @Description: 
"""
from torch import nn

from EncoderLayer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, num_encoders):
        super().__init__()
        self.enc_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads) for _ in range(num_encoders)
        ])

    def forward(self, src, src_mask):
        output = src
        for layer in self.enc_layers:
            output = layer(output, src_mask)
        return output
