_base_ = [
    'mmcls::_base_/default_runtime.py',
]

default_scope = 'mmrazor'

# !dataset config
# ==========================================================================
use_ceph = True
if use_ceph:
    file_client_args = dict(
        backend='petrel',
        path_mapping=dict({
            './data/imagenet':
            's3://openmmlab/datasets/classification/imagenet',
            'data/imagenet':
            's3://openmmlab/datasets/classification/imagenet'
        }))
else:
    file_client_args = dict(backend='disk')

dataset_type = 'mmcls.ImageNet'
preprocess_cfg = dict(
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)
data_preprocessor = dict(
    type='ImgDataPreprocessor',
    # RGB format normalization parameters
    mean=preprocess_cfg['mean'],
    std=preprocess_cfg['std'],
    # convert image from BGR to RGB
    bgr_to_rgb=preprocess_cfg['to_rgb'],
)

bgr_mean = preprocess_cfg['mean'][::-1]
bgr_std = preprocess_cfg['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='RandomResizedCrop',
        scale=224,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=7,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=236,
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(
    _scope_='mmcls',
    batch_size=256,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/train.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='RepeatAugSampler', shuffle=True),
    persistent_workers=True,
)

val_dataloader = dict(
    _scope_='mmcls',
    batch_size=256,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='meta/val.txt',
        data_prefix='val',
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
    persistent_workers=True,
)
val_evaluator = dict(_scope_='mmcls', type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# !supernet config
# ==========================================================================
supernet = dict(
    type='SearchableImageClassifier',
    backbone=dict(
        type='AttentiveMobileNet',
        first_out_channels_range=[16, 24, 8],
        last_out_channels_range=[1792, 1984, 1984 - 1792],
        dropout_stages=6,
        act_cfg=dict(type='Swish')),
    neck=dict(_scope_='mmcls', type='GlobalAveragePooling'),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=1984,
        loss=dict(
            _scope_='mmcls',
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
    train_cfg=dict(
        _scope_='mmcls',
        augments=[
            dict(type='Mixup', alpha=0.2, num_classes=1000),
            dict(type='CutMix', alpha=1.0, num_classes=1000)
        ]),
    input_resizer_cfg=dict(
        input_resizer=dict(type='DynamicInputResizer'),
        mutable_shape=dict(
            type='OneShotMutableValue',
            value_list=[[192, 192], [224, 224], [256, 256], [288, 288]],
            default_value=[224, 224])))

# !autoslim algorithm config
num_samples = 2
model = dict(
    _scope_='mmrazor',
    type='BigNAS',
    num_samples=num_samples,
    drop_prob=0.2,
    architecture=supernet,
    data_preprocessor=data_preprocessor,
    distiller=dict(
        type='ConfigurableDistiller',
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='KLDivergence', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(recorder='fc', from_student=True),
                preds_T=dict(recorder='fc', from_student=False)))),
    mutators=dict(
        channel_mutator=dict(type='BigNASChannelMutator'),
        value_mutator=dict(type='DynamicValueMutator')))

model_wrapper_cfg = dict(
    type='BigNASDDP', broadcast_buffers=False, find_unused_parameters=True)

# schedule settings
optim_wrapper = dict(
    optimizer=dict(weight_decay=0.01),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
    accumulative_counts=num_samples + 2)
# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=2048)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.0001,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=595,
        eta_min=1.0e-6,
        by_epoch=True,
        begin=5,
        end=600)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=600, val_interval=1)
val_cfg = dict(type='AutoSlimValLoop', calibrated_sample_nums=4096)
test_cfg = dict(type='AutoSlimTestLoop', calibrated_sample_nums=4096)

log_level = 'DEBUG'
