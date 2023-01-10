_base_ = ['dmcp_resnet_8xb32.py']

_base_.optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.25, momentum=0.9, weight_decay=0.0001))

custom_hooks = None

# model settings
model = _base_.model
model['fix_subnet'] = 'configs/pruning/mmcls/dmcp/DMCP_SUBNET_IMAGENET.yaml'
