# -*- coding: utf-8 -*-

# Copyright 2021 Reo Yoneyama (Nagoya University)
# based on a Quasi-Periodic Parallel WaveGAN script by Yi-Chiao Wu (Nagoya University)
# (https://github.com/bigpon/QPPWG)
# and also based on a Parallel WaveGAN script by Tomoki Hayashi (Nagoya University)
# (https://github.com/kan-bayashi/ParallelWaveGAN)
#  MIT License (https://opensource.org/licenses/MIT)

"""Unified Source-Filter GAN Modules."""

import sys
import logging
import math

import torch

from usfgan.layers import Conv1d
from usfgan.layers import Conv1d1x1
from usfgan.layers import FixedBlock
from usfgan.layers import AdaptiveBlock
from usfgan.layers import upsample
from usfgan.layers import SourceNetwork
from usfgan.layers import FilterNetwork
from usfgan.utils import pd_indexing, index_initial


class USFGANGenerator(torch.nn.Module):
    """uSFGAN Generator module."""

    def __init__(self,
                 sampling_rate,
                 hop_size,
                 in_channels,
                 out_channels,
                 blockFs,
                 cycleFs,
                 blockAs,
                 cycleAs,
                 cascade_modes,
                 residual_channels=64,
                 gate_channels=128,
                 skip_channels=64,
                 aux_channels=80,
                 aux_context_window=2,
                 upsample_params={"upsample_scales": [4, 2, 5, 2]}):
        """Initialize uSFGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of dilated convolution.
            residual_channels (int): Number of channels in residual conv.
            gate_channels (int):  Number of channels in gated conv.
            skip_channels (int): Number of channels in skip conv.
            aux_channels (int): Number of channels for auxiliary feature conv.
            aux_context_window (int): Context window size for auxiliary feature.
            dropout (float): Dropout rate. 0.0 means no dropout applied.
            bias (bool): Whether to use bias parameter in conv layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal structure.
            upsample_conditional_features (bool): Whether to use upsampling network.
            upsample_net (str): Upsampling network architecture.
            upsample_params (dict): Upsampling network parameters.

        """
        super(USFGANGenerator, self).__init__()

        torch.manual_seed(1)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window

        # define upsampling networks
        self.upsample_net_f0 = torch.nn.Upsample(scale_factor=hop_size)
        upsample_params.update({
            "aux_channels": aux_channels,
            "aux_context_window": aux_context_window,
        })
        self.upsample_net = getattr(upsample, "ConvInUpsampleNetwork")(**upsample_params)

        self.source_network = SourceNetwork(sampling_rate,
                                            in_channels,
                                            out_channels,
                                            blockFs[0],
                                            cycleFs[0],
                                            blockAs[0],
                                            cycleAs[0],
                                            cascade_modes[0],
                                            residual_channels,
                                            gate_channels,
                                            skip_channels,
                                            aux_channels,)

        self.filter_network = FilterNetwork(in_channels,
                                            out_channels,
                                            blockFs[1],
                                            cycleFs[1],
                                            blockAs[1],
                                            cycleAs[1],
                                            cascade_modes[1],
                                            residual_channels,
                                            gate_channels,
                                            skip_channels,
                                            aux_channels,)

    def forward(self, x, f, c, d):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            f (Tendor): F0 (B, C, T')
            c (Tensor): Local conditioning auxiliary features (B, C ,T').
            d (Tensor): Input pitch-dependent dilated factors (B, 1, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T)

        """
        # index initialization
        batch_index, ch_index = index_initial(x.size(0), self.residual_channels)

        # perform upsampling
        f_ = self.upsample_net_f0(f)
        assert f_.size(-1) == x.size(-1)
        c = self.upsample_net(c)
        assert c.size(-1) == x.size(-1)

        # generate source signals
        s = self.source_network(x, f_, c, d, batch_index, ch_index)

        # spectral filter
        x = self.filter_network(s, c, d, batch_index, ch_index)

        return x, s, f

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


class PWGDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 kernel_size=3,
                 layers=10,
                 conv_channels=64,
                 dilation_factor=1,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 bias=True,
                 use_weight_norm=True,
                 ):
        """Initialize Parallel WaveGAN Discriminator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Number of output channels.
            layers (int): Number of conv layers.
            conv_channels (int): Number of chnn layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool) Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
        """
        super(PWGDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor ** i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                Conv1d(conv_in_channels, conv_channels,
                       kernel_size=kernel_size, padding=padding,
                       dilation=dilation, bias=bias),
                getattr(torch.nn, nonlinear_activation)(inplace=True, **nonlinear_activation_params)
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        conv_last_layer = Conv1d(
            conv_in_channels, out_channels,
            kernel_size=kernel_size, padding=padding, bias=bias)
        self.conv_layers += [conv_last_layer]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.
        Args:
            x (Tensor): Input noise signal (B, 1, T).
        Returns:
            Tensor: Output tensor (B, 1, T)
        """
        for f in self.conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)
