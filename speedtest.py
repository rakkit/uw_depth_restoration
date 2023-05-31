import argparse
import os
import time


import numpy as np
import torch
from thop import clever_format, profile  # pip install thop

from networks import define_networks


class speedTestOptions:
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.isTrain = False

    def initialize(self,parser):
        parser.add_argument('--num_preHeat', type=int, default=5, help='input image size')
        parser.add_argument('--num_avg', type=int, default=10, help='input image size')


        parser.add_argument('--width', type=int, default=512, help='input image size')
        parser.add_argument('--height', type=int, default=320, help='input image size')

        parser.add_argument('--device', type=str, default='cpu', help='cpu or gpu to test networks')
        parser.add_argument('--network_type', type=str, default='simple', help='numbet of classes')
        parser.add_argument('--encoder_layers', type=int, default=18, help='numbet of classes')
        parser.add_argument('--t_net_type', type=str, default='simple', help='numbet of classes')
        parser.add_argument('--predict_depth', type=int, default=0, help='numbet of classes')
        parser.add_argument('--dual_transmiss_net', type=int, default=0, help='numbet of classes')
        parser.add_argument('--use_dehaze', type=int, default=1, help='numbet of classes')
        parser.add_argument('--cusNet_share_encoder', type=int, default=0, help='numbet of classes')


        parser.add_argument('--weights_init', type=str, default='None', help='numbet of classes')
        parser.add_argument('--scales', type=list, default=[0], help='numbet of classes')
        parser.add_argument('--print', type=bool, default=False, help='numbet of average time')

        return parser

    def parse(self, ):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,description="")
        parser = self.initialize(parser)
        self.parser = parser
        opt = parser.parse_args()
        self.opt = opt
        if opt.print:
            self.print_opts()
        return opt

    def print_opts(self,):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            #default = self.parser.get_default(k)
            #if v != default:
            #    comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


def evaluate_networks(model, inputTensor, device='CPU'):
    if device == 'CPU':
        start = time.time()
        model(inputTensor)
        end = time.time()
        return end-start
    elif device == 'GPU':
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        model(inputTensor)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)/1000
    else:
        return 0


def init_networks(opt):
    from networks.speed_test_net import superNet
    return {'superNet': superNet(opt)}


def test_on_gpu(networksDict, netName, opt):
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")
    networksDict[netName].eval()
    networksDict[netName].to(device)

    fps = []

    # opt.num_preHeat *=10
    # opt.num_avg *= 100

    # pre-heat
    inputTensor = torch.rand(1, 3, opt.height, opt.width).to(device)

    total = 0
    count = 0
    for num in range(0, opt.num_avg):
        inputTensor = torch.rand(1, 3, opt.height, opt.width).to(device)
        fps += [evaluate_networks(networksDict[netName], inputTensor, 'GPU')]
        count += 1

    flops, params = profile(networksDict[netName], inputs=(inputTensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    # flops = profile_macs(model, (inputTensor))
    # fps = opt.num_avg/total
    # flops, params = '0', '0'

    networksDict[netName].to('cpu')
    if opt.print:
        print('count = {:f}   time={:f}'.format(count,total))
    return fps, flops, params


def test_on_cpu(networksDict, netName, opt):
    networksDict[netName].eval()
    networksDict[netName].to('cpu')
    inputTensor = torch.rand(1, 3, opt.height, opt.width).to('cpu')

    fps = []
    # pre-heat
    for num in range(0, opt.num_preHeat):
        networksDict[netName](inputTensor)
    total = 0
    for num in range(0, opt.num_avg):
        fps += [evaluate_networks(networksDict[netName], inputTensor, 'CPU')]

    flops, params = profile(networksDict[netName], inputs=(inputTensor, ), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    # flops = profile_macs(model, (inputTensor))
    # fps = opt.num_avg/total

    return fps, flops, params


def test_networks(networksDict, verbose=False):
    records = {}
    for netName, net in networksDict.items():
        if opt.device == 'cpu':
            fps, macs, params = test_on_cpu(networksDict, netName, opt)

        elif opt.device == 'gpu':
            fps, macs, params = test_on_gpu(networksDict, netName, opt)

        fps_array = 1/np.array(fps)
        fps = fps_array.mean()
        fps_std = fps_array.std()

        records[netName] = {'macs': macs, 'fps': fps, 'fps_std': fps_std, 'params': params}
        if verbose:
            print('Network {:15s} |    FPS {:10f}    |    FPS_Std {:10f}  |   FLOPs {:s}  |    Params {:s}'.format(netName, fps, fps_std, macs, params))

    return records


if __name__ == "__main__":
    opt = speedTestOptions().parse()
    networksDict = init_networks(opt)

    records = test_networks(networksDict, opt.print)
    if opt.print:
        [print('--'*60) for _ in range(3)]
    msg = 'Finished {:4d}x{:4d} - | {:s} - {:d} '.format( \
        opt.width, opt.height, opt.device, opt.num_avg)
    print("-"*37, msg, "-"*37)
    msg = ' '*30 + 'net_type={:s}, t_net_type={:s}, predict_depth={:d}, dual_t={:d}, cusNet_share_encoder={:d}, cusNet_DH={:d}'.format(
        opt.network_type, opt.t_net_type, opt.predict_depth, opt.dual_transmiss_net, opt.cusNet_share_encoder, opt.use_dehaze)
    print(msg)
    for netName, metrics in records.items():
        print('Network {:15s} |  FLOPs {:s}    |    FPS {:10f}    |    FPS_Std {:10f}    |    Params {:s}'.format(netName,metrics['macs'],metrics['fps'],metrics['fps_std'],metrics['params'] ))
    [print('--'*60) for _ in range(1)]

'