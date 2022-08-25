# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmrazor.models import DAFLGenerator, ZSKTGenerator


def test_dafl_generator():
    dafl_generator = DAFLGenerator(
        img_size=32, latent_dim=10, hidden_channels=32)
    z_batch = torch.randn(8, 10)
    fake_img = dafl_generator(z_batch)
    assert fake_img.size() == torch.Size([8, 3, 32, 32])
    with pytest.raises(AssertionError):
        z_batch = torch.randn(8, 11)
        fake_img = dafl_generator(z_batch)
    with pytest.raises(ValueError):
        z_batch = torch.randn(8, 10, 1, 1)
        fake_img = dafl_generator(z_batch)

    fake_img = dafl_generator(batch_size=8)
    assert fake_img.size() == torch.Size([8, 3, 32, 32])

    # scale_factor = 4
    dafl_generator = DAFLGenerator(
        img_size=32, latent_dim=10, hidden_channels=32, scale_factor=4)
    z_batch = torch.randn(8, 10)
    fake_img = dafl_generator(z_batch)
    assert fake_img.size() == torch.Size([8, 3, 32, 32])

    # hidden_channels=64
    dafl_generator = DAFLGenerator(
        img_size=32, latent_dim=10, hidden_channels=64)
    z_batch = torch.randn(8, 10)
    fake_img = dafl_generator(z_batch)
    assert fake_img.size() == torch.Size([8, 3, 32, 32])

    with pytest.raises(AssertionError):
        fake_img = dafl_generator(data=None, batch_size=0)


def test_zskt_generator():
    zskt_generator = ZSKTGenerator(
        img_size=32, latent_dim=10, hidden_channels=32)
    z_batch = torch.randn(8, 10)
    fake_img = zskt_generator(z_batch)
    assert fake_img.size() == torch.Size([8, 3, 32, 32])
    with pytest.raises(AssertionError):
        z_batch = torch.randn(8, 11)
        fake_img = zskt_generator(z_batch)
    with pytest.raises(ValueError):
        z_batch = torch.randn(8, 10, 1, 1)
        fake_img = zskt_generator(z_batch)

    fake_img = zskt_generator(batch_size=8)
    assert fake_img.size() == torch.Size([8, 3, 32, 32])

    # scale_factor = 4
    zskt_generator = ZSKTGenerator(
        img_size=32, latent_dim=10, hidden_channels=32, scale_factor=4)
    z_batch = torch.randn(8, 10)
    fake_img = zskt_generator(z_batch)
    assert fake_img.size() == torch.Size([8, 3, 32, 32])

    # hidden_channels=64
    zskt_generator = ZSKTGenerator(
        img_size=32, latent_dim=10, hidden_channels=64)
    z_batch = torch.randn(8, 10)
    fake_img = zskt_generator(z_batch)
    assert fake_img.size() == torch.Size([8, 3, 32, 32])

    with pytest.raises(AssertionError):
        fake_img = zskt_generator(data=None, batch_size=0)
