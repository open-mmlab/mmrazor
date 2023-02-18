import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, scalar_tensor

import math
from typing import Callable, Dict


def groups_channels(in_channels, groups):
    if in_channels % groups == 0:
        return int(in_channels/groups), groups
    else:
        num_mul = in_channels // groups
        in_channels = groups * num_mul if num_mul > 0 else groups * (num_mul + 1)
        in_channels = in_channels / groups
        return int(in_channels), groups

def groups_out_channels(out_channels, groups):
    if out_channels % groups == 0:
        return out_channels, groups
    else:
        num_mul = out_channels // groups
        out_channels = groups * num_mul if num_mul > 0 else groups * (num_mul + 1)
        out_channels = out_channels
    return int(out_channels), groups



class MetaConv2d(nn.Conv2d, MetaConvMixin):

    def __init__(self, in_channels, out_channels, kernel_size, 
                stride, padding, dilation, groups, bias, padding_mode):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode):

        self.mutable_attrs: Dict[str, BaseMuable] = nn.ModuleDict
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size if not isinstance(kernel_size, int) \
                            else [kernel_size, kernel_size]
        self.base_oup = out_channels
        self.base_inp = in_channels

        self.groups_ = groups
        self.bias_ = True if bias is not False else False
        self.max_oup_channel = self.base_oup
        if in_channels/groups == 1:
            self.max_inp_channel = 1
        else:
            self.max_inp_channel = self.base_inp
        
        self.fc11 = nn.Linear(2, 64)
        self.fc12 = nn.Linear(64, self.max_oup_channel * self.max_inp_channel \
                              * self.kernel_size[0] * self.kernel_size[1])
        if self.bias_:
            self.fc_bias = nn.Sequential(
                                nn.Linear(2, 16),
                                nn.ReLU(),
                                nn.Linear(16, self.max_out_channel)
            )

    def forward(self, x: Tensor):

        inp, out = self.forward_inpoup()
        group_sample_num = self.base_inp / self.groups_
        group_sample_num = inp if group_sample_num > inp else group_sample_num
        groups_new = int(inp / group_sample_num) if int(inp / group_sample_num) > 0 else 1
        inp, _ = groups_channels(inp, groups_new)
        oup, _ = groups_out_channels(oup, groups_new)

        scale_tensor = torch.FloatTensor([inp/self.max_inp_channel, oup/self.max_out_channel]).to(x.device)
        fc11_out = F.relu(self.fc11(scale_tensor))

        vggconv3x3_weight = self.fc12(fc11_out).view(
                                        self.max_oup_channel,
                                        self.max_inp_channel,
                                        self.kernel_size[0],
                                        self.kernel_size[1])
        bias = None
        if self.bias_:
            bias = self.fc_bias(scale_tensor)
            bias = bias[:oup]
        
        out = F.conv2d(x, vggconv3x3_weight[:oup, :inp, :, :], 
                        bias=bias, stride=self.stride, padding=self.padding, groups=groups_new)
        return out
        
