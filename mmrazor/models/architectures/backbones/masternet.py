import os, sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch, argparse
from torch import nn
import torch.nn.functional as F

import argparse, random, logging, time
import torch
from torch import nn
import numpy as np

import global_utils
# import Masternet
# import PlainNet

# import PlainNet.
# from PlainNet import parse_cmd_options, _create_netblock_list_from_str_, basic_blocks, super_blocks
# from PlainNet import create_netblock_list_from_str, _create_netblock_list_from_str_, basic_blocks, super_blocks
# from PlainNet.basic_blocks import create_netblock_list_from_str, _create_netblock_list_from_str_, _build_netblock_list_from_str_, build_netblock_list_from_str
from PlainNet.basic_blocks import _build_netblock_list_from_str_, build_netblock_list_from_str
from PlainNet import Linear, PlainNetSuperBlockClass, ResBlock, ResBlockProj, BN

# import benchmark_network_latency
from ZeroShotProxy import compute_zen_score, compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, compute_NASWOT_score


class SuperBlock(nn.Module):
    def __init__(self, fix_subnet=None, argv=None, opt=None, num_classes=None, plainnet_struct=None, no_create=False,
                 **kwargs):
        super(SuperBlock, self).__init__()
        # self.argv = argv
        # self.opt = opt
        self.num_classes = num_classes
        self.plainnet_struct = plainnet_struct
        self.plainnet_struct_txt = fix_subnet

        if self.plainnet_struct_txt is not None:
            with open(self.plainnet_struct_txt, 'r') as fid:
                the_line = fid.readlines()[0].strip()
                self.plainnet_struct = the_line
            pass
        # self.module_opt = self.parse_cmd_options_2(self.argv)

        # if self.num_classes is None:
        #     self.num_classes = self.module_opt.num_classes

        # if self.plainnet_struct is None and self.module_opt.plainnet_struct is not None:
        #     self.plainnet_struct = self.module_opt.plainnet_struct

        # if self.plainnet_struct is None:
        #     # load structure from text file
        #     if hasattr(opt, 'plainnet_struct_txt') and opt.plainnet_struct_txt is not None:
        #         plainnet_struct_txt = opt.plainnet_struct_txt
        #     else:
        #         plainnet_struct_txt = self.module_opt.plainnet_struct_txt

        #     if plainnet_struct_txt is not None:
        #         with open(plainnet_struct_txt, 'r') as fid:
        #             the_line = fid.readlines()[0].strip()
        #             self.plainnet_struct = the_line
        #         pass

        if self.plainnet_struct is None:
            return

        the_s = self.plainnet_struct  # type: str
        # SuperConvK3BNRELU(3,32,2,1)
        # SuperResK3K3(32,64,2,32,1)
        # SuperResK3K3(64,128,2,64,1)
        # SuperResK3K3(128,256,2,128,1)
        # SuperResK3K3(256,512,2,256,1)
        # SuperConvK1BNRELU(256,512,1,1)
        block_list, remaining_s = _build_netblock_list_from_str_(the_s, no_create=no_create, **kwargs)
        # block_list, remaining_s = _create_netblock_list_from_str_(the_s, no_create=no_create, **kwargs)
        assert len(remaining_s) == 0

        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)  # register

    # def parse_cmd_options_2(self, argv, opt=None):
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument('--plainnet_struct', type=str, default=None, help='PlainNet structure string')
    #     parser.add_argument('--plainnet_struct_txt', type=str, default=None, help='PlainNet structure file name')
    #     parser.add_argument('--num_classes', type=int, default=None, help='how to prune')
    #     module_opt, _ = parser.parse_known_args(argv)

    #     return module_opt

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

class MasterNet(SuperBlock):
    def __init__(self, fix_subnet=None, argv=None, opt=None, num_classes=1000, plainnet_struct=None, no_create=False,
                 no_reslink=None, no_BN=None, use_se=None):

        # fix_subnet = opt.plainnet_struct_txt
        # no_BN = opt.no_BN
        # no_reslink = opt.no_reslink
        # use_se = opt.use_se

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

def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert hasattr(the_net, 'split')
    splitted_net_str = the_net.split(split_layer_threshold=6)
    return splitted_net_str

def get_new_random_structure_str(AnyPlainNet, structure_str, num_classes, get_search_space_func,
                                 num_replaces=1): # structure_str 初始化的网络结构 init_plainnet
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert isinstance(the_net, SuperBlock)
    selected_random_id_set = set()
    for replace_count in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)
        to_search_student_blocks_list_list = get_search_space_func(the_net.block_list, random_id)

        to_search_student_blocks_list = [x for sublist in to_search_student_blocks_list_list for x in sublist]
        new_student_block_str = random.choice(to_search_student_blocks_list)

        if len(new_student_block_str) > 0:
            # new_student_block = PlainNet.create_netblock_list_from_str(new_student_block_str, no_create=True)
            # new_student_block = create_netblock_list_from_str(new_student_block_str, no_create=True)
            new_student_block = build_netblock_list_from_str(new_student_block_str, no_create=True)
            assert len(new_student_block) == 1
            new_student_block = new_student_block[0]
            if random_id > 0:
                last_block_out_channels = the_net.block_list[random_id - 1].out_channels
                new_student_block.set_in_channels(last_block_out_channels)
            the_net.block_list[random_id] = new_student_block
        else:
            # replace with empty block
            the_net.block_list[random_id] = None
    pass  # end for

    # adjust channels and remove empty layer
    tmp_new_block_list = [x for x in the_net.block_list if x is not None]
    last_channels = the_net.block_list[0].out_channels
    for block in tmp_new_block_list[1:]:
        block.set_in_channels(last_channels)
        last_channels = block.out_channels
    the_net.block_list = tmp_new_block_list

    new_random_structure_str = the_net.split(split_layer_threshold=6)
    return new_random_structure_str

def get_latency(AnyPlainNet, random_structure_str, gpu, args):
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=False)
    if gpu is not None:
        the_model = the_model.cuda(gpu)

    def get_model_latency(model, batch_size, resolution, in_channels, gpu, repeat_times, fp16):
        if gpu is not None:
            device = torch.device('cuda:{}'.format(gpu))
        else:
            device = torch.device('cpu')

        if fp16:
            model = model.half()
            dtype = torch.float16
        else:
            dtype = torch.float32

        the_image = torch.randn(batch_size, in_channels, resolution, resolution, dtype=dtype,
                                device=device)

        model.eval()
        warmup_T = 3
        with torch.no_grad():
            for i in range(warmup_T):
                the_output = model(the_image)
            start_timer = time.time()
            for repeat_count in range(repeat_times):
                the_output = model(the_image)

        end_timer = time.time()
        the_latency = (end_timer - start_timer) / float(repeat_times) / batch_size
        return the_latency
    

    # the_latency = benchmark_network_latency.get_model_latency(model=the_model, batch_size=args.batch_size,
    the_latency = get_model_latency(
        model=the_model,
        batch_size=args.batch_size,
        resolution=args.input_image_size,
        in_channels=3, gpu=gpu, repeat_times=1,
        fp16=True)

    del the_model
    torch.cuda.empty_cache()
    return the_latency

def compute_nas_score(AnyPlainNet, random_structure_str, gpu, args):
    # compute network zero-shot proxy score
    # arch_1 = 'SuperConvK3BNRELU(3,24,2,1)SuperResK3K3(24,32,2,64,1)SuperResK5K5(32,64,2,32,1)SuperResK5K5(64,168,2,96,1)SuperResK1K5K1(168,320,1,120,1)SuperResK1K5K1(320,640,2,304,3)SuperResK1K5K1(640,512,1,384,1)SuperConvK1BNRELU(512,2384,1,1)'
    # random_structure_str = 'SuperConvK3BNRELU(3,24,2,1)SuperResK1K5K1(24,32,2,32,1)SuperResK1K7K1(32,104,2,64,1)SuperResK1K5K1(104,512,2,160,1)SuperResK1K5K1(512,344,1,192,1)SuperResK1K5K1(344,688,2,320,4)SuperResK1K5K1(688,680,1,304,3)SuperConvK1BNRELU(680,2552,1,1)'
    the_model = AnyPlainNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                            no_create=False, no_reslink=True)
    the_model = the_model.cuda(gpu)
    try:
        if args.zero_shot_score == 'Zen':
            the_nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    mixup_gamma=args.gamma, batch_size=args.batch_size,
                                                                    repeat=1)
            the_nas_core = the_nas_core_info['avg_nas_score']
        elif args.zero_shot_score == 'TE-NAS':
            the_nas_core = compute_te_nas_score.compute_NTK_score(model=the_model, gpu=gpu,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)

        elif args.zero_shot_score == 'Syncflow':
            the_nas_core = compute_syncflow_score.do_compute_nas_score(model=the_model, gpu=gpu,
                                                                       resolution=args.input_image_size,
                                                                       batch_size=args.batch_size)

        elif args.zero_shot_score == 'GradNorm':
            the_nas_core = compute_gradnorm_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                    resolution=args.input_image_size,
                                                                    batch_size=args.batch_size)

        elif args.zero_shot_score == 'Flops':
            the_nas_core = the_model.get_FLOPs(args.input_image_size)

        elif args.zero_shot_score == 'Params':
            the_nas_core = the_model.get_model_size()

        elif args.zero_shot_score == 'Random':
            the_nas_core = np.random.randn()

        elif args.zero_shot_score == 'NASWOT':
            the_nas_core = compute_NASWOT_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                  resolution=args.input_image_size,
                                                                  batch_size=args.batch_size)
    except Exception as err:
        logging.info(str(err))
        logging.info('--- Failed structure: ')
        logging.info(str(the_model))
        # raise err
        the_nas_core = -9999


    del the_model
    torch.cuda.empty_cache()
    return the_nas_core

def search_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--zero_shot_score', type=str, default=None,
                        help='could be: Zen (for Zen-NAS), TE (for TE-NAS)')
    parser.add_argument('--search_space', type=str, default=None,
                        help='.py file to specify the search space.')
    parser.add_argument('--evolution_max_iter', type=int, default=int(48e4),
                        help='max iterations of evolution.')
    parser.add_argument('--budget_model_size', type=float, default=None, help='budget of model size ( number of parameters), e.g., 1e6 means 1M params')
    parser.add_argument('--budget_flops', type=float, default=None, help='budget of flops, e.g. , 1.8e6 means 1.8 GFLOPS')
    parser.add_argument('--budget_latency', type=float, default=None, help='latency of forward inference per mini-batch, e.g., 1e-3 means 1ms.')
    parser.add_argument('--max_layers', type=int, default=None, help='max number of layers of the network.')
    parser.add_argument('--batch_size', type=int, default=None, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--population_size', type=int, default=512, help='population size of evolution.')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='output directory')
    parser.add_argument('--gamma', type=float, default=1e-2,
                        help='noise perturbation coefficient')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='number of classes')
    parser.add_argument('--plainnet_struct', type=str, default=None, help='PlainNet structure string')
    parser.add_argument('--plainnet_struct_txt', type=str, default=None, help='PlainNet structure file name')
    parser.add_argument('--no_BN', action='store_true')
    parser.add_argument('--no_reslink', action='store_true')
    parser.add_argument('--use_se', action='store_true')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


def main(args, argv):
    gpu = args.gpu
    # if gpu is not None:
    #     torch.cuda.set_device('cuda:{}'.format(gpu))
    #     torch.backends.cudnn.benchmark = True

    # 判断是否已经存在了best_structure
    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    if os.path.isfile(best_structure_txt):
        print('skip ' + best_structure_txt)
        return None

    # load search space config .py file
    select_search_space = global_utils.load_py_module_from_path(args.search_space)

    # load masternet
    fix_subnet = args.plainnet_struct_txt
    # masternet = MasterNet(fix_subnet=fix_subnet, num_classes=args.num_classes, opt=args, argv=argv, no_create=True)
    masternet = MasterNet(fix_subnet=fix_subnet, num_classes=args.num_classes, no_create=True)
    initial_structure_str = str(masternet)

    popu_structure_list = []
    popu_zero_shot_score_list = []
    popu_latency_list = []

    start_timer = time.time()
    for loop_count in range(args.evolution_max_iter): # 240
        # too many networks in the population pool, remove one with the smallest score
        while len(popu_structure_list) > args.population_size: # 512个候选网络
            min_zero_shot_score = min(popu_zero_shot_score_list)
            tmp_idx = popu_zero_shot_score_list.index(min_zero_shot_score)
            popu_zero_shot_score_list.pop(tmp_idx)
            popu_structure_list.pop(tmp_idx)
            popu_latency_list.pop(tmp_idx)
        pass

        if loop_count >= 1 and loop_count % 100 == 0:
            max_score = max(popu_zero_shot_score_list)
            min_score = min(popu_zero_shot_score_list)
            elasp_time = time.time() - start_timer
            logging.info(f'loop_count={loop_count}/{args.evolution_max_iter}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')

        # ----- generate a random structure ----- #
        if len(popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=MasterNet, structure_str=initial_structure_str, num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=1)
        else:
            tmp_idx = random.randint(0, len(popu_structure_list) - 1)
            tmp_random_structure_str = popu_structure_list[tmp_idx]
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=MasterNet, structure_str=tmp_random_structure_str, num_classes=args.num_classes,
                get_search_space_func=select_search_space.gen_search_space, num_replaces=2)

        random_structure_str = get_splitted_structure_str(MasterNet, random_structure_str,
                                                          num_classes=args.num_classes)

        the_model = None
        # 经过筛选 
        # max_layers / budget_model_size / budget_flops / budget_latency=0.0001
        if args.max_layers is not None: # 10
            if the_model is None:
                the_model = MasterNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False) # 这里去除VCNN指的是只有包含RELU激活函数的卷积层构成的网络，不包含BN、残差块等算子，并且只取到GAP(global average pool)层的前一层以保留更多的信息。
            the_layers = the_model.get_num_layers()
            if args.max_layers < the_layers:
                continue

        if args.budget_model_size is not None:
            if the_model is None:
                the_model = MasterNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_size = the_model.get_model_size()
            if args.budget_model_size < the_model_size:
                continue

        if args.budget_flops is not None:
            if the_model is None:
                the_model = MasterNet(num_classes=args.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_flops = the_model.get_FLOPs(args.input_image_size)
            if args.budget_flops < the_model_flops:
                continue

        the_latency = np.inf
        """ # latency """
        if args.budget_latency is not None: # 0.0001
            the_latency = get_latency(MasterNet, random_structure_str, gpu, args)
            if args.budget_latency < the_latency:
                continue
        
        # 计算这个score
        the_nas_core = compute_nas_score(MasterNet, random_structure_str, gpu, args)

        popu_structure_list.append(random_structure_str)
        popu_zero_shot_score_list.append(the_nas_core)
        popu_latency_list.append(the_latency)

    return popu_structure_list, popu_zero_shot_score_list, popu_latency_list


if __name__ == '__main__':
    args = search_cmd_options(sys.argv)
    log_fn = os.path.join(args.save_dir, 'evolution_search.log')
    global_utils.create_logging(log_fn)

    info = main(args, sys.argv)
    if info is None:
        exit()

    popu_structure_list, popu_zero_shot_score_list, popu_latency_list = info

    # export best structure
    best_score = max(popu_zero_shot_score_list)
    best_idx = popu_zero_shot_score_list.index(best_score)
    best_structure_str = popu_structure_list[best_idx]
    the_latency = popu_latency_list[best_idx]

    best_structure_txt = os.path.join(args.save_dir, 'best_structure.txt')
    global_utils.mkfilepath(best_structure_txt)
    with open(best_structure_txt, 'w') as fid:
        fid.write(best_structure_str)
    pass  # end with
