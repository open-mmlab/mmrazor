# TODO(shiguang): to 'mmrazor'
default_scope = 'mmcls'

# !dataset config
# ==========================================================================
use_mc = True
if use_mc:
    file_client_args = dict(
        backend='memcached',
        server_list_cfg='/mnt/lustre/share/memcached_client/server_list.conf',
        client_cfg='/mnt/lustre/share/memcached_client/client.conf')
else:
    file_client_args = dict(backend='disk')

dataset_type = 'ImageNet'
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

extra_params = dict(
    translate_const=int(224 * 0.45),
    img_mean=tuple(round(x) for x in data_preprocessor['mean']),
)
policies = [
    [
        dict(
            type='EqualizeV2',
            prob=0.8,
            magnitude=1,
            extra_params=extra_params),
        dict(type='ShearY', prob=0.8, magnitude=4, extra_params=extra_params),
    ],
    [
        dict(type='Color', prob=0.4, magnitude=9, extra_params=extra_params),
        dict(
            type='EqualizeV2',
            prob=0.6,
            magnitude=3,
            extra_params=extra_params),
    ],
    [
        dict(type='Color', prob=0.4, magnitude=1, extra_params=extra_params),
        dict(
            type='RotateV2', prob=0.6, magnitude=8, extra_params=extra_params),
    ],
    [
        dict(
            type='SolarizeV2',
            prob=0.8,
            magnitude=3,
            extra_params=extra_params),
        dict(
            type='EqualizeV2',
            prob=0.4,
            magnitude=7,
            extra_params=extra_params),
    ],
    [
        dict(
            type='SolarizeV2',
            prob=0.4,
            magnitude=2,
            extra_params=extra_params),
        dict(
            type='SolarizeV2',
            prob=0.6,
            magnitude=2,
            extra_params=extra_params),
    ],
    [
        dict(type='Color', prob=0.2, magnitude=0, extra_params=extra_params),
        dict(
            type='EqualizeV2',
            prob=0.8,
            magnitude=8,
            extra_params=extra_params),
    ],
    [
        dict(
            type='EqualizeV2',
            prob=0.4,
            magnitude=8,
            extra_params=extra_params),
        dict(
            type='SolarizeAddV2',
            prob=0.8,
            magnitude=3,
            extra_params=extra_params),
    ],
    [
        dict(type='ShearX', prob=0.2, magnitude=9, extra_params=extra_params),
        dict(
            type='RotateV2', prob=0.6, magnitude=8, extra_params=extra_params),
    ],
    [
        dict(type='Color', prob=0.6, magnitude=1, extra_params=extra_params),
        dict(
            type='EqualizeV2',
            prob=1.0,
            magnitude=2,
            extra_params=extra_params),
    ],
    [
        dict(
            type='InvertV2', prob=0.4, magnitude=9, extra_params=extra_params),
        dict(
            type='RotateV2', prob=0.6, magnitude=0, extra_params=extra_params),
    ],
    [
        dict(
            type='EqualizeV2',
            prob=1.0,
            magnitude=9,
            extra_params=extra_params),
        dict(type='ShearY', prob=0.6, magnitude=3, extra_params=extra_params),
    ],
    [
        dict(type='Color', prob=0.4, magnitude=7, extra_params=extra_params),
        dict(
            type='EqualizeV2',
            prob=0.6,
            magnitude=0,
            extra_params=extra_params),
    ],
    [
        dict(
            type='PosterizeV2',
            prob=0.4,
            magnitude=6,
            extra_params=extra_params),
        dict(
            type='AutoContrastV2',
            prob=0.4,
            magnitude=7,
            extra_params=extra_params),
    ],
    [
        dict(
            type='SolarizeV2',
            prob=0.6,
            magnitude=8,
            extra_params=extra_params),
        dict(type='Color', prob=0.6, magnitude=9, extra_params=extra_params),
    ],
    [
        dict(
            type='SolarizeV2',
            prob=0.2,
            magnitude=4,
            extra_params=extra_params),
        dict(
            type='RotateV2', prob=0.8, magnitude=9, extra_params=extra_params),
    ],
    [
        dict(
            type='RotateV2', prob=1.0, magnitude=7, extra_params=extra_params),
        dict(
            type='TranslateYRel',
            prob=0.8,
            magnitude=9,
            extra_params=extra_params),
    ],
    [
        dict(type='ShearX', prob=0.0, magnitude=0, extra_params=extra_params),
        dict(
            type='SolarizeV2',
            prob=0.8,
            magnitude=4,
            extra_params=extra_params),
    ],
    [
        dict(type='ShearY', prob=0.8, magnitude=0, extra_params=extra_params),
        dict(type='Color', prob=0.6, magnitude=4, extra_params=extra_params),
    ],
    [
        dict(type='Color', prob=1.0, magnitude=0, extra_params=extra_params),
        dict(
            type='RotateV2', prob=0.6, magnitude=2, extra_params=extra_params),
    ],
    [
        dict(
            type='EqualizeV2',
            prob=0.8,
            magnitude=4,
            extra_params=extra_params),
        dict(
            type='EqualizeV2',
            prob=0.0,
            magnitude=8,
            extra_params=extra_params),
    ],
    [
        dict(
            type='EqualizeV2',
            prob=1.0,
            magnitude=4,
            extra_params=extra_params),
        dict(
            type='AutoContrastV2',
            prob=0.6,
            magnitude=2,
            extra_params=extra_params),
    ],
    [
        dict(type='ShearY', prob=0.4, magnitude=7, extra_params=extra_params),
        dict(
            type='SolarizeAddV2',
            prob=0.6,
            magnitude=7,
            extra_params=extra_params),
    ],
    [
        dict(
            type='PosterizeV2',
            prob=0.8,
            magnitude=2,
            extra_params=extra_params),
        dict(
            type='SolarizeV2',
            prob=0.6,
            magnitude=10,
            extra_params=extra_params),
    ],
    [
        dict(
            type='SolarizeV2',
            prob=0.6,
            magnitude=8,
            extra_params=extra_params),
        dict(
            type='EqualizeV2',
            prob=0.6,
            magnitude=1,
            extra_params=extra_params),
    ],
    [
        dict(type='Color', prob=0.8, magnitude=6, extra_params=extra_params),
        dict(
            type='RotateV2', prob=0.4, magnitude=5, extra_params=extra_params),
    ],
]

train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='AutoAugmentV2', policies=policies),
    dict(type='PackClsInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(
        type='ResizeEdge',
        scale=256,
        edge='short',
        backend='pillow',
        interpolation='bilinear'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackClsInputs')
]

train_dataloader = dict(
    batch_size=64,
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
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
# ==========================================================================

supernet = dict(
    _scope_='mmrazor',
    type='SearchableImageClassifier',
    backbone=dict(
        type='AttentiveMobileNet',
        first_out_channels_range=[16, 24, 8],
        last_out_channels_range=[1792, 1984, 1984 - 1792],
        dropout_stages=6,
        norm_cfg=dict(type='DynamicBatchNorm2d', momentum=0.0),
        act_cfg=dict(type='Swish')),
    head=dict(
        type='DynamicLinearClsHead',
        num_classes=1000,
        in_channels=1984,
        loss=dict(
            type='mmcls.LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)),
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
        channel_mutator=dict(
            type='mmrazor.OneShotChannelMutator',
            channel_unit_cfg={
                'type': 'OneShotMutableChannelUnit',
                'default_args': {
                    'unit_predefined': True
                }
            },
            parse_cfg={'type': 'Predefined'}),
        value_mutator=dict(type='DynamicValueMutator')))

model_wrapper_cfg = dict(
    type='mmrazor.BigNASDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# !schedule config
# ==========================================================================
# ! grad clip by value is not supported!
optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.8, momentum=0.9, weight_decay=0.00001, nesterov=True),
    clip_grad=dict(clip_value=1.0),
    paramwise_cfg=dict(bias_decay_mult=0., norm_decay_mult=0.),
    accumulative_counts=num_samples + 2)

# learning policy
max_epochs = 360
warmup_epochs = 5
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=0.001,  # 0.0001
        by_epoch=False,
        begin=0,
        end=3125),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=max_epochs,
        eta_min=0,
        by_epoch=True,
        begin=0,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='mmrazor.AutoSlimValLoop', calibrated_sample_nums=4096)
test_cfg = dict(type='mmrazor.AutoSlimTestLoop', calibrated_sample_nums=4096)

# !runtime config
# ==========================================================================
# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(
        type='CheckpointHook', interval=1, max_keep_ckpts=10,
        save_best='auto'),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ClsVisualizer', vis_backends=vis_backends)

# load from which checkpoint
# load_from = None
load_from = '/mnt/lustre/sunyue1/autolink/workspace-547/0802_mmrazor/work_dirs/1107_train_8/epoch_360.pth'

# whether to resume training from the loaded checkpoint
resume = False

# set log level
# log_level = 'DEBUG'

# test_dataloader for nas
train_dataloader = dict(
    batch_size=64,
    # num_workers=16,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root='data/imagenet',
        ann_file='/mnt/lustre/share_data/wangshiguang/train_4k.txt',
        data_prefix='train',
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
    persistent_workers=True,
)

# SRUN_ARGS='-p pat_dev -x HOST-10-198-32-[12,22]' GPUS=32 tools/slurm_test_.sh pat_dev xxx configs/nas/mmcls/bignas/attentive_mobilenet_supernet_32xb64_in1k.py /mnt/lustre/sunyue1/autolink/workspace-547/0802_mmrazor/work_dirs/1107_train_8/epoch_360.pth --cfg-options env_cfg.dist_cfg.port=24974
