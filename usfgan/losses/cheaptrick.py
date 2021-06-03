# -*- coding: utf-8 -*-

# Copyright 2021 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Spectral envelope estimation module based on the idea of CheapTrick.
   Please see https://www.sciencedirect.com/science/article/pii/S0167639314000697 for details."""

import math

import torch
import torch.nn as nn
import torch.fft


class AdaptiveWindowing(nn.Module):

    def __init__(self,
                 sampling_rate,
                 hop_size,
                 fft_size,
                 f0_floor,
                 f0_ceil,
        ):
        """Initilize AdaptiveWindowing module.
        Args:
            sampling_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
        """
        super(AdaptiveWindowing, self).__init__()

        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.window = torch.zeros((f0_ceil+1, fft_size)).cuda()
        self.zero_padding = nn.ConstantPad2d((fft_size // 2, fft_size // 2, 0, 0), 0)

        # Pre-calculation of the window functions
        for f0 in range(f0_floor, f0_ceil + 1):
            half_win_len = round(1.5 * self.sampling_rate / f0)
            base_index = torch.arange(-half_win_len, half_win_len + 1, dtype=torch.int64)
            position = base_index / 1.5 / self.sampling_rate
            left = fft_size // 2 - half_win_len
            right = fft_size // 2 + half_win_len + 1
            window = torch.zeros(fft_size)
            window[left: right] = 0.5 * torch.cos(math.pi * position * f0) + 0.5
            average = torch.sum(window * window).pow(0.5)
            self.window[f0] = (window / average)

    def forward(self, x, f):
        """Calculate forward propagation.
        Args:
            x (Tensor): Waveform (B, fft_size // 2 + 1, T).
            f (Tensor): F0 sequence (B, T').
        Returns:
            Tensor: Power spectrogram (B, bin_size, T').
        """
        # Get the matrix of window functions corresponding to F0
        x = self.zero_padding(x).unfold(1, self.fft_size, self.hop_size)
        windows = self.window[f]
        # Adaptive windowing and calculate power spectrogram.
        # In test, change x[:, : -1, :] to x.
        x = torch.abs(torch.fft.rfft(x[:, : -1, :] * windows)).pow(2)

        return x


class AdaptiveLiftering(nn.Module):

    def __init__(self,
                 sampling_rate,
                 fft_size,
                 f0_floor,
                 f0_ceil,
                 q1=-0.15,
        ):
        """Initilize AdaptiveLiftering module.
        Args:
            sampling_rate (int): Sampling rate.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
            q1 (float): Parameter to remove effect of adjacent harmonics.
        """
        super(AdaptiveLiftering, self).__init__()

        self.sampling_rate = sampling_rate
        self.bin_size = fft_size // 2 + 1
        self.q1 = q1
        self.q0 = 1.0 - 2.0 * q1
        self.smoothing_lifter = torch.zeros((f0_ceil+1, self.bin_size)).cuda()
        self.compensation_lifter = torch.zeros((f0_ceil+1, self.bin_size)).cuda()

        # Pre-calculation of the smoothing lifters and compensation lifters
        for f0 in range(f0_floor, f0_ceil + 1):
            smoothing_lifter = torch.zeros(self.bin_size)
            compensation_lifter = torch.zeros(self.bin_size)
            quefrency = torch.arange(1, self.bin_size) / sampling_rate
            smoothing_lifter[0] = 1.0
            smoothing_lifter[1:] = torch.sin(math.pi * f0 * quefrency) / (math.pi * f0 * quefrency)
            compensation_lifter[0] = self.q0 + 2.0 * self.q1
            compensation_lifter[1:] = self.q0 + 2.0 * self.q1 * torch.cos(2.0 * math.pi * f0 * quefrency)
            self.smoothing_lifter[f0] = smoothing_lifter
            self.compensation_lifter[f0] = compensation_lifter

    def forward(self, x, f):
        """Calculate forward propagation.
        Args:
            x (Tensor): Power spectrogram (B, bin_size, T').
            f (Tensor): F0 sequence (B, T').
        Returns:
            Tensor: Estimated spectral envelope (B, bin_size, T').
        """
        # Setting the smoothing lifter and compensation lifter
        smoothing_lifter = self.smoothing_lifter[f]
        compensation_lifter = self.compensation_lifter[f]
        # Calculating cepstrum
        tmp = torch.cat((x, torch.flip(x[:, :, 1:-1], [2])), dim=2)
        cepstrum = torch.fft.rfft(
            torch.log(torch.clamp(tmp, min=1e-7))
            ).real
        # Liftering cepstrum with the lifters
        liftered_cepstrum = cepstrum * smoothing_lifter * compensation_lifter
        # Return the result to the spectral domain
        x = torch.fft.irfft(liftered_cepstrum)[:, :, : self.bin_size]

        return x


class CheapTrick(nn.Module):

    def __init__(self,
                 sampling_rate,
                 hop_size,
                 fft_size,
                 f0_floor,
                 f0_ceil,
                 uv_threshold=0,
                 q1=-0.15
        ):
        """Initilize AdaptiveLiftering module.
        Args:
            sampling_rate (int): Sampling rate.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
            fft_size (int): FFT size.
            q1 (float): Parameter to remove effect of adjacent harmonics.
        """
        super(CheapTrick, self).__init__()

        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.uv_threshold = uv_threshold

        self.ada_wind = AdaptiveWindowing(
                            sampling_rate,
                            hop_size,
                            fft_size,
                            f0_floor,
                            f0_ceil,
        )
        self.ada_lift = AdaptiveLiftering(
                            sampling_rate,
                            fft_size,
                            f0_floor,
                            f0_ceil,
                            q1,
        )

    def forward(self, x, f):
        """Calculate forward propagation.
        Args:
            x (Tensor): Power spectrogram (B, T).
            f (Tensor): F0 sequence (B, T').
        Returns:
            Tensor: Estimated spectral envelope (B, bin_size, T').
        """
        # Step0: Round F0 values to integers.
        voiced = (f > self.uv_threshold) * torch.ones_like(f)
        f = voiced * f + (1 - voiced) * self.f0_ceil
        f = torch.round(
                torch.clamp(f, min=self.f0_floor, max=self.f0_ceil)
            ).to(torch.int64)
        # Step1: Adaptive windowing and calculate power spectrogram.
        x = self.ada_wind(x, f)
        # Step3: Smoothing (log axis) and spectral recovery on the cepstrum domain.
        x = self.ada_lift(x, f)

        return x


if __name__ == "__main__":
    """Test of spectral envelope extraction."""
    import numpy as np
    import pyworld as pw
    import soundfile as sf
    import librosa.display
    import matplotlib.pyplot as plt

    config = {
        'sampling_rate': 16000,
        'hop_size': 80,
        'fft_size': 1024,
        'f0_floor': 50,
        'f0_ceil': 500
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cheaptrick = CheapTrick(**config)
    cheaptrick.to(device)

    file_name = "../../egs/arctic/data/wav/arctic_evaluation/bdl/bdl_arctic_b0474.wav"
    x, sr = sf.read(file_name)
    x = x[:config['sampling_rate']]
    _f0, t = pw.dio(x, config['sampling_rate'], frame_period=config['hop_size'] * 1000 / config['sampling_rate'])
    f0 = pw.stonemask(x, _f0, t, config['sampling_rate'])
    ap = pw.d4c(x, f0, t, config['sampling_rate'])

    x = torch.from_numpy(np.array(x[np.newaxis, :])).clone().to(device)
    f0 = torch.from_numpy(np.array(f0[np.newaxis, :])).clone().to(device)
    sp = torch.exp(cheaptrick.forward(x, f0))
    sp = sp.to('cpu').numpy().copy()[0]
    f0 = f0.to('cpu').numpy().copy()[0]

    # confirm whether the signal is resynthesized properly
    y = pw.synthesize(f0, sp, ap, config['sampling_rate'], config['hop_size'] * 1000 / config['sampling_rate'])
    save_name = 'resynthesized.wav'
    sf.write(save_name, y, config['sampling_rate'])

    # confirm whether reasonable spectral envelopes are extracted
    sp_db = librosa.power_to_db(sp)
    librosa.display.specshow(data=sp_db.T, sr=config['sampling_rate'],
                             hop_length=config['hop_size'], y_axis='linear', x_axis='time')
    plt.colorbar(format="%+2.f dB")
    save_name = 'spectrogram.png'
    plt.savefig(save_name)
