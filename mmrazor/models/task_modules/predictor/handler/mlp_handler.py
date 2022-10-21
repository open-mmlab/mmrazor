# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
from mmcv.cnn.bricks import build_activation_layer
from mmdet.models.losses import SmoothL1Loss
from mmengine.optim.scheduler import CosineAnnealingLR

from mmrazor.registry import TASK_UTILS
from .base_handler import BaseHandler


class MLP(nn.Module):
    """MLP implemented with nn.Linear.

    Input: Tensor with shape [B, C, H, W].
    Output: Tensor with shape [B, C, H, W].
    Args:
        in_features (int): Dimension of input features.
        hidden_features (int): Dimension of hidden features.
        out_features (int): Dimension of output features.
        act_cfg (dict): The config dict for activation between pointwise
            convolution. Defaults to ``dict(type='ReLU')``.
        drop (float): Dropout rate. Defaults to 0.0.
    """

    def __init__(self,
                 in_features: int = 78,
                 hidden_features: int = 300,
                 out_features: int = 1,
                 num_hidden_layers: int = 2,
                 act_cfg: Dict = dict(type='ReLU'),
                 drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = build_activation_layer(act_cfg)

        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(nn.Linear(hidden_features, hidden_features))
            hidden_layers.append(build_activation_layer(act_cfg))
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.hidden_layers(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

    @staticmethod
    def init_weights(m):
        import numpy as np
        if type(m) == nn.Linear:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.uniform_(-y, y)
            m.bias.data.fill_(0)


@TASK_UTILS.register_module()
class MLPHandler(BaseHandler):

    def __init__(self, model_cfg: Dict = None, device: str = 'cpu') -> None:
        self.model_cfg = model_cfg if model_cfg is not None else dict()
        self.model = MLP(**self.model_cfg)
        self.device = device

    def fit(self, train_data, train_label, **train_cfg):
        self.model = self.run(train_data, train_label, **train_cfg)

    def predict(self, test_data):
        """Predict candidates."""
        if test_data.ndim < 2:
            data = torch.zeros(1, test_data.shape[0])
            data[0, :] = torch.from_numpy(test_data).float()
        else:
            data = torch.from_numpy(test_data).float()

        self.model = self.model.to(device=self.device)
        self.model.eval()
        with torch.no_grad():
            data = data.to(device=self.device)
            pred = self.model(data)

        return pred.cpu().detach().numpy()

    def check_dimentions(self, shape):
        if shape != self.model.fc1.in_features:
            self.model.fc1 = nn.Linear(shape, self.model.fc1.out_features)

    def load(self, path):
        """Load predictor's pretrained weights."""
        self.model.load_state_dict(
            torch.load(path, map_location='cpu')['state_dict'])

    def save(self, path):
        """Save predictor and return saved path for diff suffix;"""
        path = path + '_mlp.pth'
        torch.save({'state_dict': self.model.state_dict(), 'meta': {}}, path)
        return path

    def run(self,
            train_data,
            train_label,
            data_split: float = 0.8,
            epoch: int = 2000):
        """Train MLP network."""
        num_samples = train_data.shape[0]
        target = torch.zeros(num_samples, 1)
        perm = torch.randperm(target.size(0))
        train_index = perm[:int(num_samples * data_split)]
        valid_index = perm[int(num_samples * data_split):]

        inputs = torch.from_numpy(train_data).float()
        target[:, 0] = torch.from_numpy(train_label).float()

        self.model = self.model.to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=8e-4)
        self.criterion = SmoothL1Loss()

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=epoch, eta_min=0, by_epoch=True)

        best_loss = 1e33
        for _ in range(epoch):
            train_inputs = inputs[train_index].to(self.device)
            train_labels = target[train_index].to(self.device)

            self.model.train()
            self.optimizer.zero_grad()
            pred = self.model(train_inputs)
            loss = self.criterion(pred, train_labels)
            loss.backward()
            self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                valid_inputs = inputs[valid_index].to(self.device)
                valid_labels = target[valid_index].to(self.device)

                pred = self.model(valid_inputs)
                valid_loss = self.criterion(pred, valid_labels).item()

            self.scheduler.step()

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_net = copy.deepcopy(self.model)

        return best_net.to(device='cpu')
