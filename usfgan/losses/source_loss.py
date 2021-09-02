# -*- coding: utf-8 -*-

# Copyright 2021 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Source Excitation Spectral Envelope Regularization Loss module."""

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
            sampling_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum F0 value.
            f0_ceil (int): Maximum F0 value.
            uv_threshold (float): V/UV determining threshold.
            q1 (float): Parameter to remove effect of adjacent harmonics.
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

    def forward(self, s, f):
        """Calculate forward propagation.
        Args:
            s (Tensor): Predicted source signal (B, T).
            f (Tensor): Extracted F0 sequence (B, T').
        Returns:
            source_loss (Tensor): Source loss value.
        """
        spectral_envelope = self.cheaptrick.forward(s, f)
        zeros = torch.zeros_like(spectral_envelope)
        source_loss = self.loss(zeros, spectral_envelope)

        return source_loss / len(s)
