# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import torch
import torch.nn as nn

from networks.p2pdehaze_utils import SELayer

# from utils.layers import Conv3x3, ConvBlock, upsample


class SimpleLightDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(SimpleLightDecoder, self).__init__()
        self.num_ch_enc = num_ch_enc[-1] if not isinstance(num_ch_enc, int) else num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        conv = [
            SELayer(self.num_ch_enc),
            nn.Conv2d(self.num_ch_enc, 256, 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            # pose ,0
            nn.Conv2d(num_input_features * 256, 256, 3, stride, 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            # pose ,1
            nn.Conv2d(256, 256, 3, stride, 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            # pose ,2
            nn.Conv2d(256, 3 * num_frames_to_predict_for, 1),
            nn.Sigmoid(),
        ]

        self.conv = nn.Sequential(*conv)

    def forward(self, input_features):

        if not isinstance(input_features, list):
            input_features = [input_features]
        out = self.conv(input_features[-1])
        out = out.mean(3).mean(2)
        return out.view(-1, 3, 1, 1)
