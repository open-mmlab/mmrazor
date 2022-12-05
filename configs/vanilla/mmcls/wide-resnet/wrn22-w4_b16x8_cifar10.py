_base_ = ['wrn16-w2_b16x8_cifar10.py']
model = dict(
    backbone=dict(depth=22, widen_factor=4), head=dict(in_channels=256, ))
