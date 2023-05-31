#!/usr/bin/python
# -*- coding: utf-8 -*-

from utils.network_utils import init_net


def define_mnist_network(opt):
    from networks.mnistNet import mnistNet
    net = mnistNet()
    return init_net(net)


def define_encoder_network(opt):
    if opt.network_type == 'custom':
        from networks.archive.customNetwork import customeEncoder
        net = customeEncoder(3)
    elif opt.network_type == 'simple':
        from networks.resnet_encoder import ResnetEncoder
        net = ResnetEncoder(opt.encoder_layers, opt.weights_init
                            == 'pretrained')
    else:
        net = None

        # raise ValueError("CANNOT FIND THE NETWORK")

    return net


def define_pose_network(opt):
    from networks.posenet import PoseDecoder
    return PoseDecoder()


def define_transmission_network(num_ch_enc, opt, t_net_type='simple'):
    if t_net_type == 'beta':
        if opt.network_type == 'zeros':
            from networks.zeros import BetaDecoder
            net = BetaDecoder()
        elif opt.network_type == 'dehaze':
            from networks.dehaze import BetaDecoder
            net = BetaDecoder()
    else:

        net = None
        raise ValueError('CANNOT FIND THE NETWORK')
    return net


def define_light_network(num_ch_enc, opt):

    if opt.network_type in ['custom', 'simple']:
        from networks.light_decoder import SimpleLightDecoder
        net = SimpleLightDecoder(num_ch_enc)
    else:
        net = None
    return net


def define_depth_network(opt):
    if opt.network_type == 'zeros':
        from networks.zeros import DepthNet
        net = DepthNet()
    elif opt.network_type == 'dehaze':
        from networks.dehaze import DepthNet
        net = DepthNet(opt.last_flag)

    return net


def define_zero_network(opt):
    if opt.network_type == 'zeros':
        from networks.zeros import MainModel
        return MainModel()


def define_dehaze_network(opt):
    if opt.network_type == 'dehaze':
        from networks.dehaze import RestoreModel
        return RestoreModel(3)
