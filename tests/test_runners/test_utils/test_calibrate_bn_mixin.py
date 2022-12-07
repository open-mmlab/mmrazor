# Copyright (c) OpenMMLab. All rights reserved.
from logging import Logger
from typing import Sequence
from unittest import TestCase

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from mmrazor.engine.runner.utils import CalibrateBNMixin


class ToyModel(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.bn = nn.BatchNorm2d(3)

    def forward(self, x: Tensor) -> Tensor:
        return self.bn(x)

    def test_step(self, x: Tensor) -> None:
        self(x)


class ToyRunner:

    def __init__(self) -> None:
        self.model = ToyModel()
        self.logger = Logger('calibrate test logger')


class ToyValLoop(CalibrateBNMixin):

    def __init__(self) -> None:
        self.fp16 = False
        self.runner = ToyRunner()


class FakeDataset(Dataset):

    def __init__(self,
                 random_nums: int = 64,
                 x_shape: Sequence[int] = (3, 224, 224)) -> None:
        self.random_x = torch.normal(1, 100, size=(random_nums, *x_shape))
        self.random_nums = random_nums
        self.x_shape = list(x_shape)

    def __getitem__(self, index: int) -> Tensor:
        return self.random_x[index]

    def __len__(self) -> int:
        return self.random_nums

    @property
    def data(self) -> Tensor:
        return self.random_x


class TestCalibrateBNMixin(TestCase):

    def test_calibrate_bn_statistics(self) -> None:
        dataloader = self.prepare_dataloader(random_nums=2000)
        loop = ToyValLoop()
        loop.calibrate_bn_statistics(dataloader, 2000)

        calibrated_data = dataloader.dataset.data
        calibrated_mean = calibrated_data.mean((0, 2, 3))
        calibrated_var = calibrated_data.var((0, 2, 3), unbiased=True)

        assert torch.allclose(calibrated_mean,
                              loop.runner.model.bn.running_mean)
        assert torch.allclose(calibrated_var, loop.runner.model.bn.running_var)

    def prepare_dataloader(
        self,
        random_nums: int = 2000,
        x_shape: Sequence[int] = (3, 224, 224)
    ) -> DataLoader:
        dataset = FakeDataset(random_nums=random_nums, x_shape=x_shape)

        return DataLoader(dataset, batch_size=64)
