# -*- coding: utf-8 -*-

# Copyright 2021 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Spectral envelope estimation module based on the idea of CheapTrick."""

import math

import torch
import torch.fft


class Step1(torch.nn.Module):

    def __init__(self,
                 sampling_rate, 
                 hop_size, 
                 f0_floor, 
                 f0_ceil, 
                 fft_size):
        super(Step1, self).__init__()
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.amp_floor = 0.000000000000001
        self.window = torch.zeros((f0_ceil+1, fft_size), dtype=torch.float64).cuda()
        self.window_weight = torch.zeros((f0_ceil+1), dtype=torch.float64).cuda()
        self.zero_padding = torch.nn.ConstantPad2d((fft_size // 2, fft_size // 2, 0, 0), 0)

        # Pre-calculation of the window functions
        for f0 in range(f0_floor, f0_ceil + 1):
            half_win_len = round(1.5 * self.sampling_rate / f0)
            base_index = torch.arange(-half_win_len, half_win_len + 1, dtype=torch.int64)
            position = torch.zeros((half_win_len * 2 + 1), dtype=torch.float64)
            position = base_index / 1.5 / self.sampling_rate

            window = torch.zeros(fft_size, dtype=torch.float64)
            left = fft_size // 2 - half_win_len
            right = fft_size // 2 + half_win_len + 1
            window[left: right] = 0.5 * torch.cos(math.pi * position * f0) + 0.5
            average = torch.sum(window * window) ** 0.5

            self.window[f0] = (window / average)
            self.window_weight[f0] = torch.sum(window)

    def forward(self, x, f0):

        # Get the matrix of window functions corresponding to F0
        x = self.zero_padding(x).unfold(1, self.fft_size, self.hop_size)
        windows = self.window[f0]

        # Adaptive windowing and calculate power spectrogram.
        # In test, change x[:, : -1, :] to x.
        x = torch.abs(torch.fft.rfft(x[:, : -1, :] * windows)).pow(2)

        return x


# class Step2(torch.nn.Module):

#     def __init__(self,
#                  sampling_rate,
#                  fft_size):
#         super(Step2, self).__init__()

#         self.sampling_rate = sampling_rate
#         self.fft_size = fft_size
#         self.bin_size = fft_size // 2 + 1
#         self.frequency_interval = sampling_rate / fft_size
#         self.eps = 0.0000000000000001

#     def forward(self, x, f0):

#         batch_size, seq_len = f0.size()
#         smoothed_x = torch.empty((batch_size, seq_len, self.bin_size), dtype=torch.float64).cuda()
#         width = f0 * 2.0 / 3.0
#         boundary = torch.round(width * self.fft_size / self.sampling_rate) + 1
#         origin_of_mirroring_axis = (boundary - 0.5) * self.sampling_rate / self.fft_size - width / 2.0
#         low_base = origin_of_mirroring_axis / self.frequency_interval
#         high_base = (origin_of_mirroring_axis + width) / self.frequency_interval

#         for i in range(batch_size):
#             for j in range(seq_len):
#                 b = int(boundary[i, j].item())
#                 l = int(low_base[i, j].item())
#                 h = int(high_base[i, j].item())
#                 x_bottom_flip = torch.flip(x[i, j, : b], [0])
#                 x_top_flip = torch.flip(x[i, j, -b: ], [0])
#                 mirroring_x = torch.cat((x_bottom_flip, x[i, j], x_top_flip), dim=0)
#                 segment = torch.cumsum(mirroring_x * self.sampling_rate / self.fft_size, dim=0)
#                 smoothed_x[i, j] = (segment[h: h + self.bin_size] 
#                                   - segment[l: l + self.bin_size]) / width[i, j]

#         return smoothed_x + self.eps
        

class Step3(torch.nn.Module):

    def __init__(self, 
                 sampling_rate, 
                 f0_floor, 
                 f0_ceil, 
                 fft_size, 
                 q1):
        super(Step3, self).__init__()

        self.sampling_rate = sampling_rate
        self.bin_size = fft_size // 2 + 1
        self.q1 = q1
        self.q0 = 1.0 - 2.0 * q1
        self.smoothing_lifter = torch.zeros((f0_ceil+1, self.bin_size), dtype=torch.float64).cuda()
        self.compensation_lifter = torch.zeros((f0_ceil+1, self.bin_size), dtype=torch.float64).cuda()

        # Pre-calculation of the smoothing lifters and compensation lifters
        for f0 in range(f0_floor, f0_ceil + 1):
            smoothing_lifter = torch.zeros(self.bin_size, dtype=torch.float64)
            compensation_lifter = torch.zeros(self.bin_size, dtype=torch.float64)

            quefrency = torch.arange(1, self.bin_size, dtype=torch.float64) /sampling_rate
            smoothing_lifter[0] = 1.0
            smoothing_lifter[1:] = torch.sin(math.pi * f0 * quefrency) / (math.pi * f0 * quefrency)
            compensation_lifter[0] = self.q0 + 2.0 * self.q1
            compensation_lifter[1:] = self.q0 + 2.0 * self.q1 * torch.cos(2.0 * math.pi * f0 * quefrency)

            self.smoothing_lifter[f0] = smoothing_lifter
            self.compensation_lifter[f0] = compensation_lifter

    def forward(self, x, f0):

        # Setting the smoothing lifter and compensation lifter
        smoothing_lifter = self.smoothing_lifter[f0]
        compensation_lifter = self.compensation_lifter[f0]

        # Calculating cepstrum
        tmp = torch.cat((x, torch.flip(x[:, :, 1:-1], [2])), dim=2)
        cepstrum = torch.fft.rfft(torch.log(tmp)).real

        # Liftering cepstrum with the lifters
        liftered_cepstrum = cepstrum * smoothing_lifter * compensation_lifter

        # Return the result to the spectral domain
        x = torch.fft.irfft(liftered_cepstrum)[:, :, : self.bin_size]

        return x


class CheapTrick(torch.nn.Module):

    def __init__(self,
                 sampling_rate,
                 hop_size,
                 fft_size,
                 f0_floor,
                 f0_ceil,
                 uv_threshold=0,
                 q1=-0.15):
        super(CheapTrick, self).__init__()

        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.uv_threshold = uv_threshold
        self.q1 = q1

        self.step1 = Step1(self.sampling_rate, 
                           self.hop_size,
                           self.f0_floor,
                           self.f0_ceil, 
                           self.fft_size)

        # self.step2 = Step2(self.sampling_rate, 
        #                    self.fft_size)

        self.step3 = Step3(self.sampling_rate, 
                           self.f0_floor, 
                           self.f0_ceil, 
                           self.fft_size,
                           self.q1)
    
    def forward(self, x, f0):

        # Step0: Round F0 values to integers.
        voiced = (f0 > self.uv_threshold) * torch.ones_like(f0)
        f0 = voiced * f0 + (1 - voiced) * self.f0_ceil
        f0 = torch.round(
                torch.clamp(f0,
                            min=self.f0_floor,
                            max=self.f0_ceil)
                        ).type(torch.int64)

        # Step1: Adaptive windowing and calculate power spectrogram.
        x = self.step1(x, f0)

        # Step2: Smoothing of the power spectrogram (linear axis). 
        # This step is ommited for faster computation.
        # x = self.step2(x, f0)
    
        # Step3: Smoothing (log axis) and spectral recovery on the cepstrum domain.
        x = self.step3(x, f0)

        return x


if __name__ == "__main__":

    """Test of spectral envelope extraction."""

    import time
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

    x = torch.from_numpy(np.array(x[np.newaxis, :]).astype(np.float64)).clone().to(device)
    f0 = torch.from_numpy(np.array(f0[np.newaxis, :]).astype(np.float64)).clone().to(device)
    sp = torch.exp(cheaptrick.forward(x, f0))
    sp = sp.to('cpu').numpy().copy().astype('float64')[0]
    f0 = f0.to('cpu').numpy().copy().astype('float64')[0]

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