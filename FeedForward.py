# -*- coding: utf-8 -*-

"""
# @Date: 2025/8/2 13:12
# @File: FeedForward.py
# @Author: 10993
# @Description: 前反馈
"""
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffn(x)
