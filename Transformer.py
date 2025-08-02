# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/1 23:22
# @File: Transformer.py
# @Author: 10993
# @Description: 
"""
import math

import torch
from torch import nn

from Decoder import Decoder
from Encoder import Encoder
from PositionalEncodeing import PositionalEncoding


class Transformer(nn.Module):

    def __init__(self, d_model=512, num_heads=8, num_encoders=6,
                 num_decoders=6, src_vocab_size=10000,
                 tgt_vocab_size=10000, max_len=5000
                 ):
        super().__init__()

        self.d_model = d_model

        # 编解码器
        self.encoder = Encoder(d_model, num_heads, num_encoders)
        self.decoder = Decoder(d_model, num_heads, num_decoders)

        # 位置信息编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Embeddings 词嵌入
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # 线性和归一层
        self.output = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_pad_token=0, tgt_pad_token=0):
        src_mask = self.create_pad_mask(src, src_pad_token)
        tgt_mask = self.create_pad_mask(tgt, tgt_pad_token)
        subsequent_mask = self.create_subsequent_mask(tgt.size(1)).to(tgt.device)
        tgt_mask = tgt_mask & subsequent_mask

        # 嵌入
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)

        # 位置编码
        src_emb = self.pos_encoding(src_emb)
        tgt_emb = self.pos_encoding(tgt_emb)

        # 编码
        enc_out = self.encoder(src_emb, src_mask)

        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask, src_mask)

        output = self.output(dec_out)
        return output

    def create_pad_mask(self, seq, pad_token):
        return (seq != pad_token).unsqueeze(1).unsqueeze(2)

    def create_subsequent_mask(self, size):
        mask = torch.tril(torch.ones(size, size)).bool()
        return mask
