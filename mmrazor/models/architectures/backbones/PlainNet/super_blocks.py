'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import uuid
import global_utils

import PlainNet
from PlainNet import _get_right_parentheses_index_, basic_blocks

class PlainNetSuperBlockClass(basic_blocks.PlainNetBasicBlockClass):
    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super(PlainNetSuperBlockClass, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sub_layers = sub_layers
        self.no_create = no_create
        self.block_list = None
        self.module_list = None

    def forward(self, x): # 
        output = x
        for block in self.block_list:
            output = block(output)
        return output

    def __str__(self):
        return type(self).__name__ + '({},{},{},{})'.format(self.in_channels, self.out_channels,
                                                                self.stride, self.sub_layers)

    def __repr__(self):
        return type(self).__name__ + '({}|{},{},{},{})'.format(self.block_name, self.in_channels, self.out_channels,
                                                               self.stride, self.sub_layers)

    def get_output_resolution(self, input_resolution):
        resolution = input_resolution
        for block in self.block_list:
            resolution = block.get_output_resolution(resolution)
        return resolution

    def get_FLOPs(self, input_resolution):
        resolution = input_resolution
        flops = 0.0
        for block in self.block_list:
            flops += block.get_FLOPs(resolution)
            resolution = block.get_output_resolution(resolution)
        return flops


    def get_model_size(self):
        model_size = 0.0
        for block in self.block_list:
            model_size += block.get_model_size()
        return model_size

    def set_in_channels(self, c):
        self.in_channels = c
        if len(self.block_list) == 0:
            self.out_channels = c
            return

        self.block_list[0].set_in_channels(c)
        last_channels = self.block_list[0].out_channels
        if len(self.block_list) >= 2 and \
                (isinstance(self.block_list[0], basic_blocks.ConvKX) or isinstance(self.block_list[0], basic_blocks.ConvDW)) and \
                isinstance(self.block_list[1], basic_blocks.BN):
            self.block_list[1].set_in_channels(last_channels)

    def encode_structure(self):
        return [self.out_channels, self.sub_layers]

    @classmethod
    def create_from_str(cls, s, no_create=False, **kwargs):
        assert cls.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len(cls.__name__ + '('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        sub_layers = int(param_str_split[3])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                       sub_layers=sub_layers, block_name=tmp_block_name, no_create=no_create,
                   **kwargs),\
               s[idx + 1:]


class SuperConvKXBNRELU(PlainNetSuperBlockClass):
    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, kernel_size=None,
                 no_create=False, no_reslink=False, no_BN=False, **kwargs):
        super(SuperConvKXBNRELU, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.sub_layers = sub_layers
        self.kernel_size = kernel_size
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN

        # if self.no_reslink:
        #     print('Warning! {} use no_reslink'.format(str(self)))
        # if self.no_BN:
        #     print('Warning! {} use no_BN'.format(str(self)))

        full_str = ''
        last_channels = in_channels
        current_stride = stride
        for i in range(self.sub_layers):
            if not self.no_BN:
                inner_str = 'ConvKX({},{},{},{})BN({})RELU({})'.format(last_channels, self.out_channels,
                                                                   self.kernel_size,
                                                                   current_stride,
                                                                   self.out_channels, self.out_channels)
            else:
                inner_str = 'ConvKX({},{},{},{})RELU({})'.format(last_channels, self.out_channels,
                                                                       self.kernel_size,
                                                                       current_stride,
                                                                       self.out_channels)
            full_str += inner_str

            last_channels = out_channels
            current_stride = 1
        pass

        self.block_list = PlainNet.create_netblock_list_from_str(full_str, no_create=no_create,
                                                                 no_reslink=no_reslink, no_BN=no_BN)
        if not no_create:
            self.module_list = nn.ModuleList(self.block_list)
        else:
            self.module_list = None

    def forward_pre_relu(self, x):
        output = x
        for block in self.block_list[0:-1]:
            output = block(output)
        return output

    def __str__(self):
        return type(self).__name__ + '({},{},{},{})'.format(self.in_channels, self.out_channels,
                                                                self.stride, self.sub_layers)

    def __repr__(self):
        return type(self).__name__ + '({}|in={},out={},stride={},sub_layers={},kernel_size={})'.format(
            self.block_name, self.in_channels, self.out_channels, self.stride, self.sub_layers, self.kernel_size)

    def split(self, split_layer_threshold):
        return str(self)

    def structure_scale(self, scale=1.0, channel_scale=None, sub_layer_scale=None):
        if channel_scale is None:
            channel_scale = scale
        if sub_layer_scale is None:
            sub_layer_scale = scale

        new_out_channels = global_utils.smart_round(self.out_channels * channel_scale)
        new_sub_layers = max(1, round(self.sub_layers * sub_layer_scale))

        return type(self).__name__ + '({},{},{},{})'.format(self.in_channels, new_out_channels,
                                                     self.stride, new_sub_layers)



class SuperConvK1BNRELU(SuperConvKXBNRELU):
    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super(SuperConvK1BNRELU, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           sub_layers=sub_layers,
                                           kernel_size=1,
                                           no_create=no_create, **kwargs)

class SuperConvK3BNRELU(SuperConvKXBNRELU):
    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super(SuperConvK3BNRELU, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           sub_layers=sub_layers,
                                           kernel_size=3,
                                           no_create=no_create, **kwargs)

class SuperConvK5BNRELU(SuperConvKXBNRELU):
    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super(SuperConvK5BNRELU, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           sub_layers=sub_layers,
                                           kernel_size=5,
                                           no_create=no_create, **kwargs)


class SuperConvK7BNRELU(SuperConvKXBNRELU):
    def __init__(self, in_channels=None, out_channels=None, stride=None, sub_layers=None, no_create=False, **kwargs):
        super(SuperConvK7BNRELU, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           sub_layers=sub_layers,
                                           kernel_size=7,
                                           no_create=no_create, **kwargs)


def register_netblocks_dict(netblocks_dict: dict):
    this_py_file_netblocks_dict = {
        'SuperConvK1BNRELU': SuperConvK1BNRELU,
        'SuperConvK3BNRELU': SuperConvK3BNRELU,
        'SuperConvK5BNRELU': SuperConvK5BNRELU,
        'SuperConvK7BNRELU': SuperConvK7BNRELU,

    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict