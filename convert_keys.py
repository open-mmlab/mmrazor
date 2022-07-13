from collections import OrderedDict

import torch
from mmengine.config import Config

from mmrazor.core import *  # noqa: F401,F403
from mmrazor.models import *  # noqa: F401,F403
from mmrazor.registry import MODELS
from mmrazor.utils import register_all_modules


def convert_spos_key(old_path, new_path):
    old_dict = torch.load(old_path)
    new_dict = {'meta': old_dict['meta'], 'state_dict': {}}

    mapping = {
        'choices': '_candidates',
        'architecture.': '',
        'model.': '',
    }

    for k, v in old_dict['state_dict'].items():
        new_key = k
        for _from, _to in mapping.items():
            new_key = new_key.replace(_from, _to)

        new_key = f'architecture.{new_key}'

        new_dict['state_dict'][new_key] = v

    torch.save(new_dict, new_path)


def convert_detnas_key(old_path, new_path):
    old_dict = torch.load(old_path)
    new_dict = {'meta': old_dict['meta'], 'state_dict': {}}

    mapping = {
        'choices': '_candidates',
        'model.': '',
    }

    for k, v in old_dict['state_dict'].items():
        new_key = k
        for _from, _to in mapping.items():
            new_key = new_key.replace(_from, _to)

        new_dict['state_dict'][new_key] = v
    torch.save(new_dict, new_path)


def convert_anglenas_key(old_path, new_path):
    old_dict = torch.load(old_path)
    new_dict = {'state_dict': {}}

    mapping = {
        'choices': '_candidates',
        'model.': '',
        'mbv2': 'mb',
    }

    for k, v in old_dict.items():
        new_key = k
        for _from, _to in mapping.items():
            new_key = new_key.replace(_from, _to)

        new_dict['state_dict'][new_key] = v
    torch.save(new_dict, new_path)


def convert_darts_key(old_path, new_path):
    old_dict = torch.load(old_path)
    new_dict = {'meta': old_dict['meta'], 'state_dict': {}}
    cfg = Config.fromfile(
        'configs/nas/darts/darts_subnet_1xb96_cifar10_2.0.py')
    # import ipdb; ipdb.set_trace()
    model = MODELS.build(cfg.model)

    print('============> module name')
    for name, module in model.state_dict().items():
        print(name)

    mapping = {
        'choices': '_candidates',
        'model.': '',
        'edges': 'route',
    }

    for k, v in old_dict['state_dict'].items():
        new_key = k
        for _from, _to in mapping.items():
            new_key = new_key.replace(_from, _to)
            # cells.0.nodes.0.edges.choices.normal_n2_p1.0.choices.sep_conv_3x3.conv1.2.weight
            splited_list = new_key.split('.')
            if len(splited_list) > 10 and splited_list[-6] == '0':
                del splited_list[-6]
                new_key = '.'.join(splited_list)
            elif len(splited_list) > 10 and splited_list[-5] == '0':
                del splited_list[-5]
                new_key = '.'.join(splited_list)

        new_dict['state_dict'][new_key] = v

    print('============> new dict')
    for key, v in new_dict['state_dict'].items():
        print(key)

    model.load_state_dict(new_dict['state_dict'], strict=True)

    torch.save(new_dict, new_path)


def convert_cream_key(old_path, new_path):

    old_dict = torch.load(old_path, map_location=torch.device('cpu'))
    new_dict = {'state_dict': {}}  # noqa: F841

    ordered_old_dict = OrderedDict(old_dict['state_dict'])

    cfg = Config.fromfile('configs/nas/cream/cream_14_subnet_mobilenet.py')
    model = MODELS.build(cfg.model)

    model_name_list = []
    model_module_list = []

    # TODO show structure of model and checkpoint
    print('=' * 30, 'the key of model')
    for k, v in model.state_dict().items():
        print(k)

    print('=' * 30, 'the key of ckpt')
    for k, v in ordered_old_dict.items():
        print(k)

    # final mapping dict
    mapping = {}

    middle_razor2cream = {  # noqa: F841
        # point-wise expansion
        'expand_conv.conv.weight': 'conv_pw.weight',
        'expand_conv.bn.weight': 'bn1.weight',
        'expand_conv.bn.bias': 'bn1.bias',
        'expand_conv.bn.running_mean': 'bn1.running_mean',
        'expand_conv.bn.running_var': 'bn1.running_var',
        'expand_conv.bn.num_batches_tracked': 'bn1.num_batches_tracked',

        # se
        'se.conv1.conv.weight': 'se.conv_reduce.weight',
        'se.conv1.conv.bias': 'se.conv_reduce.bias',
        'se.conv2.conv.weight': 'se.conv_expand.weight',
        'se.conv2.conv.bias': 'se.conv_expand.bias',

        # depth-wise conv
        'depthwise_conv.conv.weight': 'conv_dw.weight',
        'depthwise_conv.bn.weight': 'bn2.weight',
        'depthwise_conv.bn.bias': 'bn2.bias',
        'depthwise_conv.bn.running_mean': 'bn2.running_mean',
        'depthwise_conv.bn.running_var': 'bn2.running_var',
        'depthwise_conv.bn.num_batches_tracked': 'bn2.num_batches_tracked',

        # point-wise linear projection
        'linear_conv.conv.weight': 'conv_pwl.weight',
        'linear_conv.bn.weight': 'bn3.weight',
        'linear_conv.bn.bias': 'bn3.bias',
        'linear_conv.bn.running_mean': 'bn3.running_mean',
        'linear_conv.bn.running_var': 'bn3.running_var',
        'linear_conv.bn.num_batches_tracked': 'bn3.num_batches_tracked',

    }

    first_razor2cream = {
        # for first depthsepconv dw
        'conv_dw.conv.weight': 'conv_dw.weight',
        'conv_dw.bn.weight': 'bn1.weight',
        'conv_dw.bn.bias': 'bn1.bias',
        'conv_dw.bn.running_mean': 'bn1.running_mean',
        'conv_dw.bn.running_var': 'bn1.running_var',
        'conv_dw.bn.num_batches_tracked': 'bn1.num_batches_tracked',

        # for first depthsepconv pw
        'conv_pw.conv.weight': 'conv_pw.weight',
        'conv_pw.bn.weight': 'bn2.weight',
        'conv_pw.bn.bias': 'bn2.bias',
        'conv_pw.bn.running_mean': 'bn2.running_mean',
        'conv_pw.bn.running_var': 'bn2.running_var',
        'conv_pw.bn.num_batches_tracked': 'bn2.num_batches_tracked',

        # se
        'se.conv1.conv.weight': 'se.conv_reduce.weight',
        'se.conv1.conv.bias': 'se.conv_reduce.bias',
        'se.conv2.conv.weight': 'se.conv_expand.weight',
        'se.conv2.conv.bias': 'se.conv_expand.bias',
    }

    last_razor2cream = {
        # for last convbnact
        'conv2.conv.weight': 'conv.weight',
        'conv2.bn.weight': 'bn1.weight',
        'conv2.bn.bias': 'bn1.bias',
        'conv2.bn.running_mean': 'bn1.running_mean',
        'conv2.bn.running_var': 'bn1.running_var',
        'conv2.bn.num_batches_tracked': 'bn1.num_batches_tracked',
    }

    middle_cream2razor = {v: k for k, v in middle_razor2cream.items()}
    first_cream2razor = {v: k for k, v in first_razor2cream.items()}
    last_cream2razor = {v: k for k, v in last_razor2cream.items()}

    # 1. group the razor's module names
    grouped_razor_module_name = {
        'middle': {},
        'first': [],
        'last': [],
    }

    for name, module in model.state_dict().items():
        tmp_name: str = name.split(
            'backbone.')[1] if 'backbone' in name else name
        model_name_list.append(tmp_name)
        model_module_list.append(module)

        if 'conv1' in tmp_name and len(tmp_name) <= 35:
            # belong to stem conv
            grouped_razor_module_name['first'].append(name)
        elif 'head' in tmp_name:
            # belong to last linear
            grouped_razor_module_name['last'].append(name)
        else:
            # middle
            if tmp_name.startswith('layer'):
                key_of_middle = tmp_name[5:8]
                if key_of_middle not in grouped_razor_module_name['middle']:
                    grouped_razor_module_name['middle'][key_of_middle] = [name]
                else:
                    grouped_razor_module_name['middle'][key_of_middle].append(
                        name)
            elif tmp_name.startswith('conv2'):
                key_of_middle = '7.0'
                if key_of_middle not in grouped_razor_module_name['middle']:
                    grouped_razor_module_name['middle'][key_of_middle] = [name]
                else:
                    grouped_razor_module_name['middle'][key_of_middle].append(
                        name)

    # 2. group the cream's module names
    grouped_cream_module_name = {
        'middle': {},
        'first': [],
        'last': [],
    }

    for k in ordered_old_dict.keys():
        if 'classifier' in k or 'conv_head' in k:
            # last conv
            grouped_cream_module_name['last'].append(k)
        elif 'blocks' in k:
            # middle blocks
            key_of_middle = k[7:10]
            if key_of_middle not in grouped_cream_module_name['middle']:
                grouped_cream_module_name['middle'][key_of_middle] = [k]
            else:
                grouped_cream_module_name['middle'][key_of_middle].append(k)
        else:
            # first blocks
            grouped_cream_module_name['first'].append(k)

    # 4. process the first modules
    for cream_item in grouped_cream_module_name['first']:
        if 'conv_stem' in cream_item:
            # get corresponding item from razor
            for razor_item in grouped_razor_module_name['first']:
                if 'conv.weight' in razor_item:
                    mapping[cream_item] = razor_item
                    grouped_razor_module_name['first'].remove(razor_item)
                    break
        else:
            kws = cream_item.split('.')[-1]
            # get corresponding item from razor
            for razor_item in grouped_razor_module_name['first']:
                if kws in razor_item:
                    mapping[cream_item] = razor_item
                    grouped_razor_module_name['first'].remove(razor_item)

    # 5. process the last modules
    for cream_item in grouped_cream_module_name['last']:
        if 'classifier' in cream_item:
            kws = cream_item.split('.')[-1]
            for razor_item in grouped_razor_module_name['last']:
                if 'fc' in razor_item:
                    if kws in razor_item:
                        mapping[cream_item] = razor_item
                        grouped_razor_module_name['last'].remove(razor_item)
                        break

        elif 'conv_head' in cream_item:
            kws = cream_item.split('.')[-1]
            for razor_item in grouped_razor_module_name['last']:
                if 'head.conv2' in razor_item:
                    if kws in razor_item:
                        mapping[cream_item] = razor_item
                        grouped_razor_module_name['last'].remove(razor_item)

    # 6. process the middle modules
    for cream_group_id, cream_items in grouped_cream_module_name[
            'middle'].items():
        # get the corresponding group from razor
        razor_group_id: str = str(float(cream_group_id) + 1)
        razor_items: list = grouped_razor_module_name['middle'][razor_group_id]

        if int(razor_group_id[0]) == 1:
            key_cream2razor = first_cream2razor
        elif int(razor_group_id[0]) == 7:
            key_cream2razor = last_cream2razor
        else:
            key_cream2razor = middle_cream2razor

        # matching razor items and cream items
        for cream_item in cream_items:
            # traverse all of key_cream2razor
            for cream_match, razor_match in key_cream2razor.items():
                if cream_match in cream_item:
                    # traverse razor_items to get the corresponding razor name
                    for razor_item in razor_items:
                        if razor_match in razor_item:
                            mapping[cream_item] = razor_item
                            break

    print('=' * 100)
    print('length of mapping: ', len(mapping.keys()))
    for k, v in mapping.items():
        print(k, '\t=>\t', v)
    print('#' * 100)

    # TODO DELETE this print
    print('**' * 20)
    for c, cm, r, rm in zip(ordered_old_dict.keys(), ordered_old_dict.values(),
                            model_name_list, model_module_list):
        print(f'{c}: shape {cm.shape} => {r}: shape {rm.shape}')
    print('**' * 20)

    for k, v in ordered_old_dict.items():
        print(f'Mapping from {k} to {mapping[k]}......')
        new_dict['state_dict'][mapping[k]] = v

    model.load_state_dict(new_dict['state_dict'], strict=True)

    torch.save(new_dict, new_path)


if __name__ == '__main__':
    register_all_modules(True)
    # old_path = '/mnt/lustre/dongpeijie/detnas_subnet_shufflenetv2_8xb128_in1k_acc-74.08_20211223-92e9b66a.pth'  # noqa: E501
    # new_path = '/mnt/lustre/dongpeijie/detnas_subnet_shufflenetv2_8xb128_in1k_acc-74.08_20211223-92e9b66a_2.0.pth'  # noqa: E501
    # convert_spos_key(old_path, new_path)

    # old_path = '/mnt/lustre/dongpeijie/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f.pth'  # noqa: E501
    # new_path = '/mnt/lustre/dongpeijie/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_2.0.pth'  # noqa: E501
    # convert_detnas_key(old_path, new_path)

    # old_path = './data/14.pth.tar'
    # new_path = './data/14_2.0.pth'
    # old_path = '/mnt/lustre/dongpeijie/14.pth.tar'
    # new_path = '/mnt/lustre/dongpeijie/14_2.0.pth'
    # convert_cream_key(old_path, new_path)

    # old_path = '/mnt/lustre/dongpeijie/darts_subnetnet_1xb96_cifar10_acc-97.32_20211222-e5727921.pth'  # noqa: E501
    # new_path = '/mnt/lustre/dongpeijie/darts_subnetnet_1xb96_cifar10_acc-97.32_20211222-e5727921_2.0.pth'  # noqa: E501
    # convert_darts_key(old_path, new_path)

    old_path = '/mnt/lustre/dongpeijie/spos_angelnas_flops_0.49G_acc_75.98_20220307-54f4698f.pth'  # noqa: E501
    new_path = '/mnt/lustre/dongpeijie/spos_angelnas_flops_0.49G_acc_75.98_20220307-54f4698f_2.0.pth'  # noqa: E501
    convert_anglenas_key(old_path, new_path)
