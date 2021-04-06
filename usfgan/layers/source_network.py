# -*- coding: utf-8 -*-

# Copyright 2021 Reo Yoneyama (Nagoya University)

"""Source Network module."""

import math
import logging

import torch

from usfgan.layers import Conv1d1x1
from usfgan.layers import FixedBlock
from usfgan.layers import AdaptiveBlock
from usfgan.utils import pd_indexing


class SourceNetwork(torch.nn.Module):

    def __init__(self,
                 sampling_rate,
                 in_channels,
                 out_channels,
                 blockF,
                 cycleF,
                 blockA,
                 cycleA,
                 cascade_mode,
                 residual_channels, 
                 gate_channels,
                 skip_channels,
                 aux_channels):
        super(SourceNetwork, self).__init__()

        self.sampling_rate = sampling_rate

        # convert sine signal to hidden representation
        self.conv_first = Conv1d1x1(in_channels * 2, residual_channels, bias=True)

        # check the number of blocks and cycles
        cycleA = max(cycleA, 1)
        cycleF = max(cycleF, 1)
        assert blockF % cycleF == 0
        blockF_per_cycle = blockF // cycleF
        assert blockA % cycleA == 0
        self.blockA_per_cycle = blockA // cycleA

        # define fixed residual blocks
        fixed_blocks = torch.nn.ModuleList()
        for block in range(blockF):
            dilation = 2 ** (block % blockF_per_cycle)
            conv = FixedBlock(
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                bias=True,
            )
            fixed_blocks += [conv]

        # define adaptive residual blocks
        adaptive_blocks = torch.nn.ModuleList()
        for block in range(blockA):
            conv = AdaptiveBlock(
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                bias=True,
            )
            adaptive_blocks += [conv]

        # define cascaded structure
        if cascade_mode == 0:  # fixed->adaptive
            self.conv_dilated = fixed_blocks.extend(adaptive_blocks)
            self.block_modes = [False] * blockF + [True] * blockA
        elif cascade_mode == 1:  # adaptive->fixed
            self.conv_dilated = adaptive_blocks.extend(fixed_blocks)
            self.block_modes = [True] * blockA + [False] * blockF
        else:
            logging.error("Cascaded mode %d is not supported!" % (cascade_mode))
            sys.exit(0)

        # convert hidden representation to harmonic signal
        self.conv_last = torch.nn.ModuleList([
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(skip_channels, skip_channels, bias=True),
            torch.nn.ReLU(inplace=True),
            Conv1d1x1(skip_channels, out_channels, bias=True),
        ])

    def forward(self, x, f0, c, d, batch_index, ch_index):

        with torch.no_grad():
            uv = (f0 > 0) * torch.ones_like(f0)
            rad_values = (f0 / self.sampling_rate) % 1
            b = uv * torch.sin(torch.cumsum(rad_values, dim=2) * 2 * math.pi)

        # encode to hidden representation
        x = torch.cat((x, b), dim=1)
        x = self.conv_first(x)
        
        skips = 0
        blockA_idx = 0
        for f, mode in zip(self.conv_dilated, self.block_modes):
            if mode:  # adaptive block
                dilation = 2 ** (blockA_idx % self.blockA_per_cycle)
                xP, xF = pd_indexing(x, d, dilation, batch_index, ch_index)
                x, h = f(x, xP, xF, c)
                blockA_idx += 1
            else:  # fixed block
                x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_dilated))

        # apply final layers
        x = skips
        for f in self.conv_last:
            x = f(x)

        return x