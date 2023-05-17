# Copyright (c) OpenMMLab. All rights reserved.
# model settings
import os.path as osp

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mmrazor.implementations.quantization.gptq import (GPTQCompressor,
                                                       GPTQLinear)
from mmrazor.utils import print_log


def enable_observer_linear(model):
    print_log('Enable updating qparams for GPTQLinear!')
    for _, module in model.named_modules():
        if isinstance(module, GPTQLinear):
            module.fix_qparams = False


def disable_observer_linear(model):
    print_log('Disable updating qparams for GPTQLinear!')
    for _, module in model.named_modules():
        if isinstance(module, GPTQLinear):
            module.fix_qparams = True


def get_dataloaders(batch_size, n_workers, path=''):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        osp.join(path, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
    )

    test_dataset = datasets.ImageFolder(
        osp.join(path, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]),
    )

    dataloader_train = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )
    dataloader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )
    return dataloader_train, dataloader_test


@torch.no_grad()
def eval(model: nn.Module,
         dataloader_test: DataLoader,
         device=torch.device('cuda:0'),
         is_half=True):

    total = 0
    correct = 0

    model.eval()
    with torch.no_grad():
        for x, y in dataloader_test:
            x: torch.Tensor  # type: ignore
            y: torch.Tensor  # type: ignore
            x = x.to(device)
            y = y.to(device)
            if is_half:
                x = x.half()
                y = y.half()
            outputs = model(x)
            _, predicted = outputs.max(1)
            correct += (y == predicted).long().sum()
            total += y.numel()
    acc = correct / total
    return acc


@torch.no_grad()
def infer(model: nn.Module,
          dataloader: torch.utils.data.DataLoader,
          num_samples=256,
          device=torch.device('cuda:0'),
          is_half=True):
    model.eval()
    with torch.no_grad():
        accumulate_batch = 0
        for x, _ in dataloader:
            x = x.to(device)
            if is_half:
                x = x.half()
            model(x)
            B = x.shape[0]
            accumulate_batch += B
            if accumulate_batch > num_samples:
                break


if __name__ == '__main__':
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        '--data',
        type=str,
        default='data/imagenet_torch',
        help='path to imagenet in torch folder format')
    arg_parser.add_argument(
        '--num_samples',
        type=int,
        default=512,
        help='number of samples to estimate hessian matrix')
    arg_parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='batch size for evaluation and inference')
    arg_parser.add_argument(
        '--fp16',
        type=bool,
        default=False,
        help='whether to use fp16 for evaluation and inference')
    args = arg_parser.parse_args()

    data_path = args.data
    num_samples = args.num_samples
    batch_size = args.batch_size

    model = torchvision.models.resnet18(pretrained=True)
    if args.fp16:
        model = model.half()
    train_loader, test_loader = get_dataloaders(batch_size, 4, data_path)

    compressor = GPTQCompressor()

    # # use_triton_ops is True
    # compressor.prepare(model,
    #                    quant_conv=True,
    #                    quant_linear=True,
    #                    use_triton_ops=False,
    #                    skipped_layers=['conv1'],
    #                    bits=4,
    #                    groupsize=128)

    # # quantize activation for linear
    # a_qconfig = dict(bits=4, perchannel=True, sym=False)
    compressor.prepare(
        model,
        quant_conv=True,
        quant_linear=True,
        use_triton_ops=False,
        skipped_layers=['conv1'],
        # a_qconfig=a_qconfig
    )

    model.cuda()

    enable_observer_linear(model)
    compressor.init_hessian()
    compressor.register_hessian_hooks()
    infer(model, test_loader, num_samples=num_samples, is_half=args.fp16)
    compressor.remove_hessian_hooks()
    compressor.quant_with_default_qconfig()

    print('start evaluation')
    disable_observer_linear(model)
    model = model.cuda()
    acc = eval(model, test_loader, is_half=args.fp16)
    print('accuracy:', acc.item())
