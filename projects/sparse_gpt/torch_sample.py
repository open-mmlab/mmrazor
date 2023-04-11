# model settings
import os.path as osp

import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pipe import sparse_model
from torch.utils.data import DataLoader


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


def eval(model: nn.Module, dataloader_test: DataLoader):

    total = 0
    correct = 0

    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for x, y in dataloader_test:
            x: torch.Tensor  # type: ignore
            y: torch.Tensor  # type: ignore
            x = x.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            y = y.to(device)
            correct += (y == predicted).long().sum()
            total += y.numel()
    acc = correct / total
    return acc


# sparse_model(model, train_loader, 512)

if __name__ == '__main__':
    # sparse_model(model, train_loader, 512)
    model = torchvision.models.resnet18(pretrained=True)
    train_loader, test_loader = get_dataloaders(128, 4, 'data/imagenet_torch')

    model = model.cuda()
    model = sparse_model(model, test_loader, num_batchs=512)

    print('start evaluation')
    model = model.cuda()
    acc = eval(model, test_loader)
    print('accuracy:', acc.item())
