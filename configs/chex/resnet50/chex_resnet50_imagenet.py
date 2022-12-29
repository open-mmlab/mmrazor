_base_ = ['mmcls::resnet/resnet50_8xb32_in1k.py']

data_preprocessor = {'type': 'mmcls.ClsDataPreprocessor'}
architecture = _base_.model
architecture.update({
    'init_cfg': {
        'type':
        'Pretrained',
        'checkpoint':
        'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa
    }
})
architecture.update({
    'train_cfg': dict(augments=dict(type='Mixup', alpha=0.2))
})
architecture.update({
    'head': 
        dict(
            type='LinearClsHead',
            num_classes=1000,
            in_channels=2048,
            loss=dict(
                type='LabelSmoothLoss',
                label_smooth_val=0.1,
                num_classes=1000,
                reduction='mean',
                loss_weight=1.0),
            topk=(1, 5))
})

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='ChexAlgorithm',
    architecture=architecture,
    mutator_cfg=dict(
        type='ChexMutator',
        channel_unit_cfg=dict(
            type='ChexUnit', default_args=dict(choice_mode='number', )),
        channel_ratio=0.7,
    ),
    delta_t=2,
    total_steps=180,
    init_growth_rate=0.3,
)
custom_hooks = [{'type': 'mmrazor.ChexHook'}]

train_dataloader = dict(batch_size=128)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=1.024, momentum=0.875, weight_decay=3.0517578125e-05))

# learning policy
warmup_length = 8
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1/warmup_length,
        by_epoch=True,
        begin=0,
        end=warmup_length,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        # T_max=95,
        by_epoch=True,
        begin=warmup_length,
        end=250,
    )
]

# train setting
train_cfg = dict(by_epoch=True, max_epochs=250, val_interval=1)

default_hooks = dict(
    # TODO: reset it to 100.
    # print log every 10 iterations to quickly debug.
    logger=dict(type='LoggerHook', interval=10)
)
