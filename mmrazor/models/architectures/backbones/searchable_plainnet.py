import torch
from torch import nn
import torch.nn.functional as F
from mmrazor.models.architectures.backbones.PlainNet import Linear, PlainNetSuperBlockClass, ResBlock, ResBlockProj, BN
from mmrazor.models.architectures.backbones.PlainNet.basic_blocks import _build_netblock_list_from_str_, build_netblock_list_from_str

from mmrazor.registry import MODELS


class SuperBlock(nn.Module):
    def __init__(self, fix_subnet=None, argv=None, opt=None, num_classes=None, plainnet_struct=None, no_create=False,
                 **kwargs):
        super(SuperBlock, self).__init__()
        self.argv = argv
        self.opt = opt
        self.num_classes = num_classes
        self.plainnet_struct = plainnet_struct
        self.plainnet_struct_txt = fix_subnet

        if self.plainnet_struct_txt is not None:
            with open(self.plainnet_struct_txt, 'r') as fid:
                the_line = fid.readlines()[0].strip()
                self.plainnet_struct = the_line
            pass

        if self.plainnet_struct is None:
            return

        the_s = self.plainnet_struct  # type: str
        block_list, remaining_s = _build_netblock_list_from_str_(the_s, no_create=no_create, **kwargs)
        # block_list, remaining_s = _create_netblock_list_from_str_(the_s, no_create=no_create, **kwargs)
        assert len(remaining_s) == 0

        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)  # register

    def forward(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def __str__(self):
        s = ''
        for the_block in self.block_list:
            s += str(the_block)
        return s

    def __repr__(self):
        return str(self)

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block
        if block_id < len(self.block_list):
            self.block_list[block_id + 1].set_in_channels(new_block.out_channels)

        self.module_list = nn.Module(self.block_list)

@MODELS.register_module()
class MasterNet(SuperBlock):
    def __init__(self, fix_subnet=None, argv=None, opt=None, num_classes=1000, plainnet_struct=None, no_create=False,
                 no_reslink=None, no_BN=None, use_se=None):

        super().__init__(fix_subnet=fix_subnet, argv=argv, opt=opt, num_classes=num_classes, plainnet_struct=plainnet_struct,
                                       no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se)

        self.last_channels = self.block_list[-1].out_channels
        self.fc_linear = Linear(in_channels=self.last_channels, out_channels=self.num_classes, no_create=no_create)

        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        # bn eps
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3

    def extract_stage_features_and_logit(self, x, target_downsample_ratio=None):
        stage_features_list = []
        image_size = x.shape[2]
        output = x

        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            dowsample_ratio = round(image_size / output.shape[2])
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
            pass
        pass

        output = F.adaptive_avg_pool2d(output, output_size=1)
        output = torch.flatten(output, 1)
        logit = self.fc_linear(output)
        return stage_features_list, logit

    def forward(self, x):
        output = x
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)

        output = F.adaptive_avg_pool2d(output, output_size=1)

        output = torch.flatten(output, 1)
        output = self.fc_linear(output)
        return output

    def forward_pre_GAP(self, x):
        output = x # torch.Size([64, 3, 224, 224])
        for the_block in self.block_list:
            output = the_block(output)
        return output # torch.Size([64, 512, 7, 7])

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        the_flops += self.fc_linear.get_FLOPs(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        the_size += self.fc_linear.get_model_size()

        return the_size

    def get_num_layers(self):
        num_layers = 0
        for block in self.block_list:
            assert isinstance(block, PlainNetSuperBlockClass)
            num_layers += block.sub_layers
        return num_layers

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block

        if block_id < len(self.block_list) - 1:
            if self.block_list[block_id + 1].in_channels != new_block.out_channels:
                self.block_list[block_id + 1].set_in_channels(new_block.out_channels)
        else:
            assert block_id == len(self.block_list) - 1
            self.last_channels = self.block_list[-1].out_channels
            if self.fc_linear.in_channels != self.last_channels:
                self.fc_linear.set_in_channels(self.last_channels)

        self.module_list = nn.ModuleList(self.block_list)

    def split(self, split_layer_threshold):
        new_str = ''
        for block in self.block_list:
            new_str += block.split(split_layer_threshold=split_layer_threshold)
        return new_str

    def init_parameters(self):

        for m in self.modules(): # 176
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=3.26033)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                pass

        for superblock in self.block_list:
            if not isinstance(superblock, PlainNetSuperBlockClass):
                continue
            for block in superblock.block_list:
                if not (isinstance(block, ResBlock) or isinstance(block, ResBlockProj)):
                    continue
                # print('---debug set bn weight zero in resblock {}:{}'.format(superblock, block))
                last_bn_block = None
                for inner_resblock in block.block_list:
                    if isinstance(inner_resblock, BN):
                        last_bn_block = inner_resblock
                    pass
                pass  # end for
                assert last_bn_block is not None
                # print('-------- last_bn_block={}'.format(last_bn_block))
                nn.init.zeros_(last_bn_block.netblock.weight)
        """
        i = 0
        for superblock in self.block_list:
            print('\n-start-{}'.format(superblock))
            if not isinstance(superblock, super_blocks.PlainNetSuperBlockClass):
                print('i-', i)
                continue
            j = 0
            for block in superblock.block_list:
                if not (isinstance(block, basic_blocks.ResBlock) or isinstance(block, basic_blocks.ResBlockProj)):
                    print('--j', j)
                    continue
                print('---debug set bn weight zero in resblock {}:{}'.format(superblock, block))
                last_bn_block = None
                for inner_resblock in block.block_list:
                    if isinstance(inner_resblock, basic_blocks.BN):
                        last_bn_block = inner_resblock
                    pass
                pass  # end for
                assert last_bn_block is not None
                print('-------- last_bn_block={}'.format(last_bn_block))
                nn.init.zeros_(last_bn_block.netblock.weight)
                print('--j', j)
                j = j + 1
            print('i', i)
            print('-end-{}\n'.format(superblock))
            i = i + 1
        """