'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''



import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn
import numpy as np
import global_utils, argparse, time


def network_weight_gaussian_init(net: nn.Module):
    init_true=[] # 137
    init_false = [] # 233
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
                init_true.append(m._get_name())
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                init_true.append(m._get_name())
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
                init_true.append(m._get_name())
            else:
                init_false.append(m._get_name())
                continue
    # print(init_true, init_false)
    return net

# ['Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Conv2d', 'BatchNorm2d', 'Linear'] 
# ['MasterNet', 'ModuleList', 'SuperConvK3BNRELU', 'ModuleList', 'ConvKX', 'BN', 'RELU', 'SuperResK1K5K1', 'ModuleList', 'ConvKX', 'BN', 'RELU', 'ConvKX', 'BN', 'RELU', 'ConvKX', 'BN', 'RELU', 'ConvKX', 'BN', 'RELU', 'ConvKX', 'BN', 'RELU', 'ConvKX', 'BN', 'RELU', 'SuperResK3K3', 'ModuleList', 'ConvKX', 'BN', 'RELU', 'ConvKX', 'BN', 'RELU', 'SuperResK3K3', 'ModuleList', 'ConvKX', 'BN', 'RELU', 'ConvKX', 'BN', 'RELU', 'SuperResK3K3', 'ModuleList', 'ConvKX', 'BN', 'RELU', 'ConvKX', 'BN', 'RELU', 'SuperConvK1BNRELU', 'ModuleList', 'ConvKX', 'BN', 'RELU', 'Linear']


# def network_weight_gaussian_init(net: nn.Module):
#     with torch.no_grad():
#         for m in net.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.ones_(m.weight)
#                 nn.init.zeros_(m.bias)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight)
#                 if hasattr(m, 'bias') and m.bias is not None:
#                     nn.init.zeros_(m.bias)
#             else:
#                 continue

#     return net

def compute_nas_score(gpu, model, mixup_gamma, resolution, batch_size, repeat, fp16=False):
    info = {}
    nas_score_list = []
    if gpu is not None:
        device = torch.device('cuda:{}'.format(gpu))
    else:
        device = torch.device('cpu')

    if fp16:
        dtype = torch.half
    else:
        dtype = torch.float32

    with torch.no_grad():
        for repeat_count in range(repeat):
            network_weight_gaussian_init(model)
            input = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype) # torch.mean(input) -> tensor(7.1894e-05, device='cuda:0')
            input2 = torch.randn(size=[batch_size, 3, resolution, resolution], device=device, dtype=dtype)
            mixup_input = input + mixup_gamma * input2 # mixup_gamma=0.01 torch.Size([64, 3, 224, 224])
            output = model.forward_pre_GAP(input) # torch.Size([64, 512, 7, 7]) # 
            mixup_output = model.forward_pre_GAP(mixup_input) # 给了0.01的输入扰动
            # 0.2 -> torch.Size([64, 2552, 7, 7]) / 
            nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3]) # 输出绝对值
            nas_score = torch.mean(nas_score) # tensor(2535.5146, device='cuda:0')
            # 0.2 -> 64366.5820
            # compute BN scaling
            log_bn_scaling_factor = 0.0
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                    log_bn_scaling_factor += torch.log(bn_scaling_factor)
                pass
            pass # 0.2 -> log(64366.5820) + 129.9849 = 141
            nas_score = torch.log(nas_score) + log_bn_scaling_factor # tensor(42.1751, device='cuda:0')
            nas_score_list.append(float(nas_score))

            # 一共68层BN
            # index = 0
            # log_bn_scaling_factor = 0.0
            # for m in model.modules():
            #     if isinstance(m, nn.BatchNorm2d):                    
            #         bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
            #         log_bn_scaling_factor += torch.log(bn_scaling_factor)
            #         print('{}: {}->{}->{}, sum:{}'.format(index, torch.mean(m.running_var), bn_scaling_factor, torch.log(bn_scaling_factor), log_bn_scaling_factor))
            #         index = index + 1


    std_nas_score = np.std(nas_score_list)
    avg_precision = 1.96 * std_nas_score / np.sqrt(len(nas_score_list))
    avg_nas_score = np.mean(nas_score_list) # 42.175086975097656


    info['avg_nas_score'] = float(avg_nas_score)
    info['std_nas_score'] = float(std_nas_score)
    info['avg_precision'] = float(avg_precision)
    return info


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in one mini-batch.')
    parser.add_argument('--input_image_size', type=int, default=None,
                        help='resolution of input image, usually 32 for CIFAR and 224 for ImageNet.')
    parser.add_argument('--repeat_times', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=None)
    parser.add_argument('--mixup_gamma', type=float, default=1e-2)
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

if __name__ == "__main__":
    import ModelLoader
    opt = global_utils.parse_cmd_options(sys.argv)
    args = parse_cmd_options(sys.argv)
    the_model = ModelLoader.get_model(opt, sys.argv)
    if args.gpu is not None:
        the_model = the_model.cuda(args.gpu)


    start_timer = time.time()
    info = compute_nas_score(gpu=args.gpu, model=the_model, mixup_gamma=args.mixup_gamma,
                             resolution=args.input_image_size, batch_size=args.batch_size, repeat=args.repeat_times, fp16=False)
    time_cost = (time.time() - start_timer) / args.repeat_times
    zen_score = info['avg_nas_score']
    print(f'zen-score={zen_score:.4g}, time cost={time_cost:.4g} second(s)')