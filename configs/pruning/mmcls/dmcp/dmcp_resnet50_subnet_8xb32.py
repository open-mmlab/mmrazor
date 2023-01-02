_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.25, momentum=0.9, weight_decay=0.0001))

param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[30, 60, 90], gamma=0.1)

train_cfg = dict(by_epoch=True, max_epochs=120, val_interval=1)

data_preprocessor = {'type': 'mmcls.ClsDataPreprocessor'}

# model settings
model = dict(
    _scope_='mmrazor',
    type='DMCP',
    architecture=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=False),
    mutator_cfg=dict(
        type='DMCPChannelMutator',
        channel_unit_cfg=dict(
            type='DMCPChannelUnit', default_args=dict(choice_mode='number')),
        parse_cfg=dict(
            type='BackwardTracer',
            loss_calculator=dict(type='ImageClassifierPseudoLoss'))),
    fix_subnet='configs/pruning/mmcls/dmcp/DMCP_SUBNET_IMAGENET.yaml')

model_wrapper_cfg = dict(
    type='mmrazor.DMCPDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)