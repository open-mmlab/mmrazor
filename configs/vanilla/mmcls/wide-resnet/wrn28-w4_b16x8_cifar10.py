_base_ = ['wrn16_2_b16x8_cifar10.py']
model = dict(
    backbone=dict(depth=28, widen_factor=4), head=dict(in_channels=256, ))
