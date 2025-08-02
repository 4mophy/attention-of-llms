# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/2 03:12
# @File: Decoder.py
# @Author: 10993
# @Description: 
"""
from torch import nn

from DecoderLayer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, num_decoders):
        super().__init__()
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads) for _ in range(num_decoders)
        ])

    def forward(self, tgt, enc, tgt_mask, enc_mask):
        output = tgt
        for layer in self.dec_layers:
            output = layer(output, enc, tgt_mask, enc_mask)
        return output
