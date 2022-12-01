import os
import sys
import time
import numpy as np
from tqdm import tqdm

import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx, fuse_fx
import torch.nn as nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms

# Set up warnings
import warnings
warnings.filterwarnings(
    action='ignore',
    category=DeprecationWarning,
    module=r'.*'
)
warnings.filterwarnings(
    action='default',
    module=r'torch.ao.quantization'
)

# Specify random seed for repeatable results
_ = torch.manual_seed(191009)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def evaluate(model, criterion, data_loader, use_cuda=False):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    cnt = 0
    with torch.no_grad():
        for image, target in data_loader:
            if use_cuda:
                image = image.cuda()
                target = target.cuda()
            output = model(image)
            loss = criterion(output, target)
            cnt += 1
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))
    print('')

    return top1, top5

def print_size_of_model(model):
    if isinstance(model, torch.jit.RecursiveScriptModule):
        torch.jit.save(model, "temp.p")
    else:
        torch.jit.save(torch.jit.script(model), "temp.p")
    print("Size (MB):", os.path.getsize("temp.p")/1e6)
    os.remove("temp.p")

def prepare_data_loaders(data_path, batch_size=4):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(data_path, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(data_path, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

data_path = '/nvme/dataset/cifar10/'
saved_model_dir = '../experiments/valid_convert'
os.makedirs(saved_model_dir, exist_ok=True)

_, data_loader_test = prepare_data_loaders(data_path)
criterion = nn.CrossEntropyLoss()
weights = ResNet18_Weights.DEFAULT
float_model = resnet18(weights=weights).cuda()
float_model.eval()


# deepcopy the model since we need to keep the original model around
import copy
model_to_quantize = copy.deepcopy(float_model)
model_to_quantize.eval()
qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}
prepared_model = prepare_fx(model_to_quantize, qconfig_dict)
# print(prepared_model.graph)

def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in tqdm(data_loader):
            model(image.cuda())
calibrate(prepared_model, data_loader_test)

quantized_model = convert_fx(prepared_model)
# print(quantized_model)

print("Size of model before quantization")
print_size_of_model(float_model)
print("Size of model after quantization")
print_size_of_model(quantized_model)
top1, top5 = evaluate(float_model, criterion, data_loader_test, use_cuda=True)
print("[float model] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print("[before serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))

fx_graph_mode_model_file_path = os.path.join(saved_model_dir, "resnet18_fx_graph_mode_quantized.pth")

# save with state_dict
torch.save(quantized_model.state_dict(), fx_graph_mode_model_file_path)
import copy
model_to_quantize = copy.deepcopy(float_model)
prepared_model = prepare_fx(model_to_quantize, {"": qconfig})
loaded_quantized_model = convert_fx(prepared_model)
loaded_quantized_model.load_state_dict(torch.load(fx_graph_mode_model_file_path))
top1, top5 = evaluate(loaded_quantized_model, criterion, data_loader_test)
print("[after serilaization] Evaluation accuracy on test dataset: %2.2f, %2.2f"%(top1.avg, top5.avg))
print('pipeline finish')