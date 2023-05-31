#!/usr/bin/python
# -*- coding: utf-8 -*-

# https://github.com/ErinChen1/EPDN/
# from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.p2pdehaze_utils import (Dehaze, GlobalGenerator, ResnetBlock,
                                      SELayer)


class RestoreModel(nn.Module):

    def __init__(self, input_nc):
        super(RestoreModel, self).__init__()
        self.encoder = customeEncoder(input_nc)
        self.trans_decoder = customeDecoder(None, output_nc=3,
                net_type='trans', use_dehaze=1)
        self.light_decoder = SimpleLightDecoder(self.encoder.num_ch_enc)

    def forward(self, x):
        features = self.encoder(x)
        trans = self.trans_decoder(features, None, x)
        light = self.light_decoder(features)
        return (trans, light)


class DepthNet(nn.Module):

    def __init__(self, last_flag=0):
        super(DepthNet, self).__init__()
        self.last_flag = last_flag
        self.depth_encoder = customeEncoder(3)
        self.depth_decoder = customeDecoder(None, output_nc=1,
                                            net_type='depth', last_flag=last_flag, use_dehaze=1)

    def forward(self, x):
        features = self.depth_encoder(x)
        x = self.depth_decoder(features, None, x)

        if self.last_flag == 0:
            x = 1 / x
        elif self.last_flag == 4:
        # elif self.last_flag in [1, 2, 3]:
        #     return x
            x = 1 / (10 * x + 0.05)
        return (features, x)


class customeEncoder(nn.Module):

    def __init__(
        self,
        input_nc,
        output_nc=0,
        ngf=32,
        n_downsample_global=3,
        n_blocks_global=9,
        n_local_enhancers=1,
        n_blocks_local=3,
        norm_layer=nn.InstanceNorm2d,
        padding_type='reflect',
        ):

        super(customeEncoder, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        self.downsample = {}

        # local enhancer layers #####

        for n in range(1, n_local_enhancers + 1):

            # ## downsample

            ngf_global = ngf * 2 ** (n_local_enhancers - n)
            self.downsample[n] = nn.Sequential(*[
                nn.ReflectionPad2d(3),
                nn.Conv2d(input_nc, ngf_global, kernel_size=7,
                          padding=0),
                norm_layer(ngf_global),
                nn.ReLU(True),
                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3,
                          stride=2, padding=1),
                norm_layer(ngf_global * 2),
                nn.ReLU(True),
                ])

        self.downsample_decoder = \
            nn.ModuleList(list(self.downsample.values()))

        # global generator model #####

        self.num_ch_enc = ngf_global * 2
        ngf_global = ngf * 2 ** n_local_enhancers
        self.input_downsample = nn.AvgPool2d(3, stride=2, padding=[1,
                1], count_include_pad=False)
        model_global = GlobalGenerator(
            input_nc,
            self.num_ch_enc,
            ngf_global,
            n_downsample_global,
            n_blocks_global,
            norm_layer,
            ).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        # self.dehaze=Dehaze()
        # self.dehaze2=Dehaze()

    def forward(self, input):

        # ## create input pyramid

        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.input_downsample(input_downsampled[-1]))

        # ## output at coarest level

        output_prev = self.model(input_downsampled[-1])

        # # build up one layer at a time

        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            input_i = input_downsampled[self.n_local_enhancers
                    - n_local_enhancers]
            output_prev = self.downsample[n_local_enhancers](input_i) \
                + output_prev

        return output_prev


class customeDecoder(nn.Module):

    def __init__(
        self,
        input_nc,
        output_nc,
        ngf=32,
        n_downsample_global=3,
        n_blocks_global=9,
        n_local_enhancers=1,
        n_blocks_local=3,
        norm_layer=nn.InstanceNorm2d,
        padding_type='reflect',
        use_dehaze=0,
        use_senet=0,
        last_flag=0,
        net_type='trans',
        ):

        super(customeDecoder, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        self.last_flag = last_flag
        if net_type == 'depth':
            self.net_type = 'depth'
        else:
            self.net_type = 'trans'

        for n in range(1, n_local_enhancers + 1):
            ngf_global = ngf * 2 ** (n_local_enhancers - n)
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2,
                                   padding_type=padding_type,
                                   norm_layer=norm_layer)]

            model_upsample += [nn.ConvTranspose2d(
                ngf_global * 2,
                ngf_global,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                ), norm_layer(ngf_global), nn.ReLU(True)]

            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3),
                                   nn.Conv2d(ngf, output_nc,
                                   kernel_size=7, padding=0), nn.Tanh()]

                                   # nn.Sigmoid(),

        self.deocder = nn.Sequential(*model_upsample)
        self.output_nc = output_nc
        self.use_dehaze = use_dehaze
        if self.use_dehaze:
            self.dehaze = Dehaze(3 + output_nc, output_nc, last_flag,
                                 self.net_type)
            self.dehaze2 = Dehaze(output_nc + output_nc, output_nc,
                                  last_flag, self.net_type)
        self.use_senet = use_senet
        if self.use_senet:
            self.senet = SELayer(ngf * 2 ** (n_local_enhancers - n) * 2)

    def forward(
        self,
        x,
        depth=None,
        input_x=None,
        ):
        if self.use_senet:
            x = self.senet(x)
        x = self.deocder(x)
        if self.use_dehaze:
            dehazed = torch.cat((x, input_x), 1)
            dehazed = self.dehaze(dehazed)
            x = torch.cat((x, dehazed), 1)
            x = self.dehaze2(x)

        if self.net_type != 'depth':
            x = x.expand_as(input_x)
            return x
        else:
            return x

class BetaDecoder(nn.Module):

    def __init__(self):
        super(BetaDecoder, self).__init__()

        self.InConv = SKConv(outfeatures=64, infeatures=1, M=3, L=32)

        # self.convt = DoubleConv(64, 64)

        self.OutConv = nn.Conv2d(
            64,
            3,
            3,
            padding=1,
            stride=1,
            bias=False,
            padding_mode='reflect',
            )

        # self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.conv1 = InDoubleConv(3, 64)

        self.conv2 = DoubleConv(64, 64)
        self.maxpool = nn.MaxPool2d(15, 7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(64, 12, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x, depth):

        xmin = self.InConv(x)
        atm = self.conv1(x)
        out = torch.mul(atm, xmin)
        out = self.pool(self.conv2(self.maxpool(out)))
        out = out.view(-1, 64)
        out = self.dense(out)
        out = self.relu(out)


        (n, c, h, w) = depth.shape
        copy_depth = torch.cat([depth] * 3, 1).view(n, 3, -1)

        out = out.view(-1, 3, 4)

        res = out[:, :, 0:1] * torch.exp(-out[:, :, 1:2] * copy_depth) + out[:, :, 2:3] * torch.exp(-out[:, :, 3:4] * copy_depth)
        res = res.reshape(n, 3, h, w)

        res = torch.exp(-res * depth)
        return (res, out)


class SimpleLightDecoder(nn.Module):

    def __init__(
        self,
        num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=1,
        stride=1,
        ):

        super(SimpleLightDecoder, self).__init__()
        self.num_ch_enc = (num_ch_enc[-1] if not isinstance(num_ch_enc,
                           int) else num_ch_enc)
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        conv = [  # pose ,0
                  # pose ,1
                  # pose ,2
            SELayer(self.num_ch_enc),
            nn.Conv2d(self.num_ch_enc, 256, 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(num_input_features * 256, 256, 3, stride, 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 256, 3, stride, 1),
            nn.BatchNorm2d(256),
            nn.ELU(),
            nn.Conv2d(256, 3 * num_frames_to_predict_for, 1),
            nn.Sigmoid(),
            ]

        self.conv = nn.Sequential(*conv)

    def forward(self, input_features):

        # out = self.senet(input_features[-1])
        # out = self.relu(self.convs["squeeze"](out))
        # out = self.convs[("pose", i)](out)

        if not isinstance(input_features, list):
            input_features = [input_features]
        out = self.conv(input_features[-1])

        # out = self.sigmoid(out)

        out = out.mean(3).mean(2)
        return out.view(-1, 3, 1, 1)


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                3,
                padding=1,
                bias=False,
                padding_mode='reflect',
                ),
            nn.GroupNorm(num_channels=out_ch, num_groups=8,
                         affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False,
                padding_mode='reflect',
                ),
            nn.GroupNorm(num_channels=out_ch, num_groups=8,
                         affine=True),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class InDoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                9,
                stride=4,
                padding=4,
                bias=False,
                padding_mode='reflect',
                ),
            nn.GroupNorm(num_channels=out_ch, num_groups=8,
                         affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_ch,
                out_ch,
                3,
                padding=1,
                bias=False,
                padding_mode='reflect',
                ),
            nn.GroupNorm(num_channels=out_ch, num_groups=8,
                         affine=True),
            nn.ReLU(inplace=True),
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(
            1,
            64,
            7,
            stride=4,
            padding=3,
            bias=False,
            padding_mode='reflect',
            ), nn.GroupNorm(num_channels=64, num_groups=8,
                            affine=True), nn.ReLU(inplace=True))
        self.convf = nn.Sequential(nn.Conv2d(
            64,
            64,
            3,
            padding=1,
            bias=False,
            padding_mode='reflect',
            ), nn.GroupNorm(num_channels=64, num_groups=8,
                            affine=True), nn.ReLU(inplace=True))

    def forward(self, x):
        R = x[:, 0:1, :, :]
        G = x[:, 1:2, :, :]
        B = x[:, 2:3, :, :]
        xR = torch.unsqueeze(self.conv(R), 1)
        xG = torch.unsqueeze(self.conv(G), 1)
        xB = torch.unsqueeze(self.conv(B), 1)
        x = torch.cat([xR, xG, xB], 1)
        (x, _) = torch.min(x, dim=1)
        return self.convf(x)


class SKConv(nn.Module):

    def __init__(
        self,
        outfeatures=64,
        infeatures=1,
        M=4,
        L=32,
        ):

        super(SKConv, self).__init__()
        self.M = M
        self.convs = nn.ModuleList([])
        in_conv = InConv(in_ch=infeatures, out_ch=outfeatures)
        for i in range(M):
            if i == 0:
                self.convs.append(in_conv)
            else:
                self.convs.append(nn.Sequential(nn.Upsample(scale_factor=1
                                  / 2 ** i, mode='bilinear',
                                  align_corners=True), in_conv,
                                  nn.Upsample(scale_factor=2 ** i,
                                  mode='bilinear', align_corners=True)))
        self.fc = nn.Linear(outfeatures, L)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(L, outfeatures))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for (i, conv) in enumerate(self.convs):

            # fea = conv(x).unsqueeze_(dim=1)

            fea = torch.unsqueeze(conv(x), 1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for (i, fc) in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors,
                        vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = \
            attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v
