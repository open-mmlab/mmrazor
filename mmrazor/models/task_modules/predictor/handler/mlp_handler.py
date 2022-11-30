# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mmcv.cnn.bricks import build_activation_layer

try:
    from mmdet.models.losses import SmoothL1Loss
except ImportError:
    from mmrazor.utils import get_placeholder
    SmoothL1Loss = get_placeholder('mmdet')
from mmengine.model import BaseModule
from mmengine.optim.scheduler import CosineAnnealingLR

from mmrazor.registry import TASK_UTILS
from .base_handler import BaseHandler


class MLP(BaseModule):
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
        self.init_weights()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.hidden_layers(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


@TASK_UTILS.register_module()
class MLPHandler(BaseHandler):
    """MLP handler of the metric predictor. It uses MLP network to predict the
    metric of a trained model.

    Args:
        epochs (int, optional): num of epochs for MLP network training.
            Defaults to 100.
        data_split_ratio (float, optional): split ratio of train/valid of
            input data. Defaults to 0.8.
        model_cfg (dict, optional): configs for MLP network. Defaults to None.
        device (str, optional): device for MLP Handler. Defaults to 'cuda'.
    """

    def __init__(self,
                 epochs: int = 100,
                 data_split_ratio: float = 0.8,
                 model_cfg: Dict = None,
                 device: str = 'cpu'):
        self.epochs = epochs
        self.data_split_ratio = data_split_ratio

        self.model_cfg = model_cfg if model_cfg is not None else dict()
        self.model = MLP(**self.model_cfg)

        self.device = device

    def fit(self, train_data: np.array, train_label: np.array) -> None:
        """Training the model of handler.

        Args:
            train_data (numpy.array): input data for training.
            train_label (numpy.array): input label for training.
        """
        if train_data.shape[1] != self.model.fc1.in_features:
            self.model.fc1 = nn.Linear(train_data.shape[1],
                                       self.model.fc1.out_features)
        self.model = self.train_mlp(train_data, train_label)

    def predict(self, test_data: np.array) -> np.array:
        """Predict the evaluation metric of the model.

        Args:
            test_data (numpy.array): input data for testing.

        Returns:
            numpy.array: predicted metric.
        """
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

    def load(self, path: str) -> None:
        """Load predictor's pretrained weights."""
        self.model.load_state_dict(
            torch.load(path, map_location='cpu')['state_dict'])

    def save(self, path: str) -> str:
        """Save predictor and return saved path for diff suffix."""
        path = path + '_mlp.pth'
        torch.save({'state_dict': self.model.state_dict(), 'meta': {}}, path)
        return path

    def train_mlp(self, train_data: np.array,
                  train_label: np.array) -> nn.Module:
        """Train MLP network.

        Args:
            train_data (numpy.array): input data for training.
            train_label (numpy.array): input label for training.

        Returns:
            nn.Module: the well-trained MLP network.
        """
        num_samples = train_data.shape[0]
        target = torch.zeros(num_samples, 1)
        perm = torch.randperm(target.size(0))
        train_index = perm[:int(num_samples * self.data_split_ratio)]
        valid_index = perm[int(num_samples * self.data_split_ratio):]

        inputs = torch.from_numpy(train_data).float()
        target[:, 0] = torch.from_numpy(train_label).float()

        self.model = self.model.to(device=self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=8e-4)
        self.criterion = SmoothL1Loss()

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=self.epochs, eta_min=0, by_epoch=True)

        best_loss = 1e33
        for _ in range(self.epochs):
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
