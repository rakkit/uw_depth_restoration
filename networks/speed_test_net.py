
from __future__ import absolute_import, division, print_function

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from utils.layers import Conv3x3, ConvBlock, upsample

from networks import define_networks


class superNet(nn.Module):
    def __init__(self, opt):
        super(superNet, self).__init__()
        self.cusNet_share_encoder = opt.cusNet_share_encoder
        self.opt = opt
        nets = {}
        nets['encoder'] = define_networks.define_encoder_network(opt)
        nets['net_MT'] = define_networks.define_transmission_network(nets['encoder'].num_ch_enc, opt, t_net_type=opt.t_net_type)
        nets['net_MA'] = define_networks.define_light_network(nets['encoder'].num_ch_enc, opt)

        if opt.predict_depth or opt.t_net_type == 'beta':
            nets['depth_decoder'] = define_networks.define_depth_network(nets['encoder'].num_ch_enc, opt)

        if opt.predict_depth and opt.dual_transmiss_net:
            if not self.cusNet_share_encoder:
                nets['depth_encoder'] = define_networks.define_encoder_network(opt)
            nets['net_MT'] = define_networks.define_transmission_network(nets['encoder'].num_ch_enc, opt, t_net_type='simple')
        self.nets = nets
        print(self.nets.keys())
        self.net = nn.ModuleList(list(nets.values()))


    def forward(self, x):
        if self.opt.predict_depth and self.opt.dual_transmiss_net:
            if self.cusNet_share_encoder:
                features = self.nets['encoder'](x,)
                depth = self.nets['depth_decoder'](features, None, x)
                t = self.nets['net_MT'](features, None, x)
            else:
                features = self.nets['depth_encoder'](x,)
                depth = self.nets['depth_decoder'](features, None, x)
                features = self.nets['encoder'](x)
                t = self.nets['net_MT'](features, None, x)
            a = self.nets['net_MA'](features)
        else:
            features = self.nets['encoder'](x)
            if 'depth_decoder' in self.nets.keys():
                depth = self.nets['depth_decoder'](features, None, x)
                t = self.nets['net_MT'](features, depth, x)
            else:
                t = self.nets['net_MT'](features, None, x)
            a = self.nets['net_MA'](features)
        return a
