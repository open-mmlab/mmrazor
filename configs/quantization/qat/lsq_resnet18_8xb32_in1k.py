_base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py']

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QATEpochBasedLoop',
    max_epochs=_base_.train_cfg.max_epochs)

resnet = _base_.model
ckpt = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'  # noqa: E501
resnet.init_cfg = dict(type='Pretrained', checkpoint=ckpt)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GeneralQuant',
    # data_preprocessor = dict(
    #     num_classes=1000,
    #     # RGB format normalization parameters
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     # convert image from BGR to RGB
    #     to_rgb=True,
    # ),
    architecture=resnet,
    quantizer=dict(
        type='CustomQuantizer',
        skipped_methods=[
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ],
        qconfig=dict(
            qtype='affine',
            w_observer=dict(type='mmrazor.MinMaxObserver'),
            a_observer=dict(type='mmrazor.EMAMinMaxObserver'),
            w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
            a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
            w_qscheme=dict(
                bit=8,
                is_symmetry=False,
                is_per_channel=False,
                is_pot_scale=False,
            ),
            a_qscheme=dict(
                bit=8,
                is_symmetry=False,
                is_per_channel=False,
                is_pot_scale=False),
        )))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=100,
    by_epoch=True,
    begin=0,
    end=100)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        max_keep_ckpts=3,
        out_dir='/mnt/petrelfs/caoweihan.p/training_ckpt/quant'))

model_wrapper_cfg = dict(
    type='mmrazor.GeneralQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=False)

val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')
test_cfg = val_cfg
