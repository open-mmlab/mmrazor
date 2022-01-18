_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs128_colorjittor.py',
    '../../_base_/schedules/mmcls/imagenet_bs1024_spos.py',
    '../../_base_/mmcls_runtime.py'
]
norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',
    backbone=dict(
        type='SearchableShuffleNetV2', widen_factor=1.0, norm_cfg=norm_cfg),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),
    ),
)

mutator = dict(
    type='OneShotMutator',
    placeholder_mapping=dict(
        all_blocks=dict(
            type='OneShotOP',
            choices=dict(
                shuffle_3x3=dict(
                    type='ShuffleBlock', kernel_size=3, norm_cfg=norm_cfg),
                shuffle_5x5=dict(
                    type='ShuffleBlock', kernel_size=5, norm_cfg=norm_cfg),
                shuffle_7x7=dict(
                    type='ShuffleBlock', kernel_size=7, norm_cfg=norm_cfg),
                shuffle_xception=dict(
                    type='ShuffleXception', norm_cfg=norm_cfg),
            ))))

algorithm = dict(
    type='SPOS',
    architecture=dict(
        type='MMClsArchitecture',
        model=model,
    ),
    mutator=mutator,
    distiller=None,
    retraining=False,
)

runner = dict(max_iters=150000)
evaluation = dict(interval=1000, metric='accuracy')

# checkpoint saving
checkpoint_config = dict(interval=1000)

find_unused_parameters = True
