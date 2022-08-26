_base_ = ['wrn16_2_b16x8_cifar10.py']
model = dict(
    backbone=dict(depth=40, widen_factor=2), head=dict(in_channels=128, ))
