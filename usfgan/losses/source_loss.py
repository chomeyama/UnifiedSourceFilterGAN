# -*- coding: utf-8 -*-

# Copyright 2021 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""CheapTrick-based Sourse Loss module."""

import numpy as np
import pyworld

import torch
import torch.fft
import torch.nn as nn

from usfgan.losses import CheapTrick


class SourceLoss(torch.nn.Module):

    def __init__(self,
                 sampling_rate,
                 hop_size,
                 fft_size,
                 f0_floor,
                 f0_ceil,
                 uv_threshold=0,
                 q1=-0.15):
        """Initialize source loss module.
        Args:
            fft_size (int): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (str): Window function type.
        """
        super(SourceLoss, self).__init__()

        self.cheaptrick = CheapTrick(sampling_rate=sampling_rate,
                                     hop_size=hop_size,
                                     fft_size=fft_size,
                                     f0_floor=f0_floor,
                                     f0_ceil=f0_ceil,
                                     uv_threshold=uv_threshold,
                                     q1=q1)
        self.loss = nn.MSELoss()

    def forward(self, x, f):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted source signal (B, T).
            f (Tensor): Extracted F0 sequence (B, T').
        Returns:
            loss (Tensor): Source loss value.
        """
        spectral_envelope = self.cheaptrick.forward(x, f)
        zeros = torch.zeros_like(spectral_envelope)
        loss = self.loss(zeros, spectral_envelope)

        return loss