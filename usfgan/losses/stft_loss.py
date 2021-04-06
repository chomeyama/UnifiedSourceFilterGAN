# -*- coding: utf-8 -*-

# Copyright 2021 Reo Yoneyama (Nagoya University)
# based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size, hop_size, win_length, window):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = getattr(torch, window)(win_length).cuda()
        self.amp_floor = 0.00001
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Logarithmic power STFT loss value.
        """

        x_stft = torch.stft(x, self.fft_size, self.hop_size, self.win_length,
                            window=self.window, onesided=True, pad_mode="constant")
        y_stft = torch.stft(y, self.fft_size, self.hop_size, self.win_length,
                            window=self.window, onesided=True, pad_mode="constant")
        
        x_log_pow = torch.log(torch.norm(x_stft, 2, -1).pow(2) + self.amp_floor)
        y_log_pow = torch.log(torch.norm(y_stft, 2, -1).pow(2) + self.amp_floor)

        stft_loss = self.mse_loss(x_log_pow, y_log_pow)

        return stft_loss


class MultiResolutionSTFTLoss(torch.nn.Module):

    def __init__(self,
                 fft_sizes=[512, 128, 2048],
                 hop_sizes=[80, 40, 640],
                 win_lengths=[320, 80, 1920],
                 window="hann_window"):
        """Initialize source loss module.
        Args:
            fft_sizes (int): FFT size.
            hop_sizes (int): Hop size.
            win_lengths (int): Window length.
            window (str): Window function type.
        """
        super(MultiResolutionSTFTLoss, self).__init__()

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = window
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)

        self.stft_losses = torch.nn.ModuleList()
        for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, hs, wl, window)]

    def forward(self, x, y):

        stft_loss = 0.0
        # multi resolution stft loss
        for f in self.stft_losses:
            l = f(x, y)
            stft_loss += l
        stft_loss /= len(self.stft_losses)

        return stft_loss