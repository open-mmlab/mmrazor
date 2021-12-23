# Tutorial 2: Learn about Configs

We use python files as our config system. You can find all the provided configs under `$MMRazor/configs`.

## Config Name Style

We follow the below convention to name config files. Contributors are advised to follow the same style. The config file names are divided into four parts: algorithm info, module information, training information and data information. Logically, different parts are concatenated by underscores.

`'_'`, and words in the same part are concatenated by dashes `'-'`.

```text
{algorithm info}_{model info}_[experiment setting]_{training info}_{data info}.py
```

`{xxx}` is required field and `[yyy]` is optional.

- `algorithm info`: algorithm information, algorithm name, such as spos, autoslim, cwd, etc.;

- `model info`: model information, model name to be slimmed, such as shufflenet, faster rcnn, etc;

- `experiment setting`: optional, it is used to describe important information about algorithm or model, such as there are 3 stages in spos: pre-training supernet, search, retrain subnet,  you can use it to specify which stage, you also can use it to specify teacher network and student network in KD;

- `training info`: Training information, some training schedule, including batch size, lr schedule, data augment and the like;

- `data info`: Data information, dataset name, input size and so on, such as imagenet, cifar, etc.

## Config System

Same as MMDetection, we incorporate modular and inheritance design into our config system, which is convenient to conduct various experiments.

To help the users have a basic idea of a complete config and the modules in a generation system, we make brief comments on the configs of some examples as the following. For more detailed usage and the corresponding alternative for each modules, please refer to the API documentation.

### An example of NAS - spos

```python
_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs128_colorjittor.py',     # data
    '../../_base_/schedules/mmcls/imagenet_bs1024_spos.py',   # training schedule
    '../../_base_/mmcls_runtime.py'                           # runtime setting
]

# need to specify some parameters baesd on _base_ by rewriting
evaluation = dict(interval=1000, metric='accuracy')
checkpoint_config = dict(interval=10000)
find_unused_parameters = True

# model settings
norm_cfg = dict(type='BN')
model = dict(
    type='mmcls.ImageClassifier',                   # Classifier name
    backbone=dict(
        type='SearchableShuffleNetV2',              # Backbones name
        widen_factor=1.0,
        norm_cfg=norm_cfg),
    neck=dict(type='GlobalAveragePooling'),         # neck network name
    head=dict(
        type='LinearClsHead',                       # linear classification head
        num_classes=1000,                           # The number of output categories, consistent with the number of categories in the dataset
        in_channels=1024,                           # The number of input channels, consistent with the output channel of the neck
        loss=dict(                                  # Loss function configuration information
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5),                                 # Evaluation index, Top-k accuracy rate, here is the accuracy rate of top1 and top5
    ),
)

# mutator settings
mutator = dict(
    type='OneShotMutator',                            # mutator name registered
    placeholder_mapping=dict(                         # specify mapping dict for placeholders in the architecture
        all_blocks=dict(                              # key: placeholder block name according to the architecture; value: specify mutable to replace the placeholder
            type='OneShotOP',                         # mutable name registered
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

# algorithm settings
algorithm = dict(
    type='SPOS',                                       # algorithm name registered
    architecture=dict(                                 # architecture setting
        type='MMClsArchitecture',                      # architecture name registered
        model=model,                                   # specify defined model to use in the architecture
    ),
    mutator=mutator,                                   # specify defined mutator to use in the algorithm
    distiller=None,                                    # specify the distiller in the algorithm, default None
    retraining=False,                                  # Bool, specify which stage in the algorithm. True: sunet retrain; False: pre-training supernet
)
```

### An example of KD - cwd

```python
_base_ = [
    '../../_base_/datasets/mmseg/cityscapes.py',       # data
    '../../_base_/schedules/mmseg/schedule_80k.py',    # training schedule
    '../../_base_/mmseg_runtime.py'                    # runtime setting
]

# specify norm_cfg for teacher and student as follows
norm_cfg = dict(type='SyncBN', requires_grad=True)

# pspnet r18 as student network, for more detailed usage, please refer to MMSegmentation's docs
student = dict(
    type='mmseg.EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://resnet18_v1c'),
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# pspnet r101 as teacher network, for more detailed usage, please refer to MMSegmentation's docs
teacher = dict(
    type='mmseg.EncoderDecoder',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

# distiller settings
distiller=dict(
    type='SingleTeacherDistiller',                   # distiller name registered
    teacher=teacher,                                 # specify defined teacher to use in the distiller
    teacher_trainable=False,                         # whether to train teacher
    components=[                                     # specify what moudules to calculate kd-loss in teacher and student
        dict(
            student_module='decode_head.conv_seg',   # student module name
            teacher_module='decode_head.conv_seg',   # teacher module name
            losses=[                                 # specify kd-loss
                dict(
                    type='ChannelWiseDivergence',    # kd-loss type
                    name='loss_cwd_logits',          # name this loss in order to easy get the output of this loss
                    tau=5,                           # temperature coefficient
                    loss_weight=3,                        # weight of this loss
                )
            ])
    ]),


# algorithm settings
algorithm = dict(
    type='Distillation',                                # algorithm name registered
    architecture=dict(                                  # architecture setting
        type='MMSegArchitecture',                       # architecture name registered
        model=student,                                  # specify defined student as the model of architecture
    ),
    use_gt=True,                                        # whether to calculate gt_loss with gt
    distiller=distiller,                                # specify defined distiller to use in the algorithm
)
```

### An example of pruning - autoslim

```python
_base_ = [
    '../../_base_/datasets/mmcls/imagenet_bs256_autoslim.py',   # data
    '../../_base_/schedules/mmcls/imagenet_bs2048_autoslim.py', # training schedule
    '../../_base_/mmcls_runtime.py'                             # runtime setting
]

# need to specify some parameters baesd on _base_ by rewriting
runner = dict(type='EpochBasedRunner', max_epochs=50)

# model settings
model = dict(
    type='mmcls.ImageClassifier',                            # Classifier name
    backbone=dict(type='MobileNetV2', widen_factor=1.5),     # Backbones name
    neck=dict(type='GlobalAveragePooling'),                  # neck network name
    head=dict(
        type='LinearClsHead',                                # linear classification head
        num_classes=1000,                           # The number of output categories, consistent with the number of categories in the dataset
        in_channels=1920,                           # The number of input channels, consistent with the output channel of the neck
        loss=dict(                                  # Loss function configuration information
            type='CrossEntropyLoss',
            loss_weight=1.0),
        topk=(1, 5),                                # Evaluation index, Top-k accuracy rate, here is the accuracy rate of top1 and top5
    ))

# distiller settings, for more details, please refer to the previous section: an example of KD - cwd
distiller = dict(
    type='SelfDistiller',
    components=[
        dict(
            student_module='head.fc',
            teacher_module='head.fc',
            losses=[
                dict(
                    type='KLDivergence',
                    name='loss_kd',
                    tau=1,
                    loss_weight=1,
                )
            ]),
    ])

# pruner settings
pruner=dict(
    type='RatioPruner',                         # pruner name registered
    ratios=(2 / 12, 3 / 12, 4 / 12, 5 / 12,     # list, specify the ratio range of random sampling
            6 / 12, 7 / 12, 8 / 12, 9 / 12,
            10 / 12, 11 / 12, 1.0))

# algorithm settings
algorithm = dict(
    type='AutoSlim',                            # algorithm name registered
    architecture=dict(                          # architecture setting
        type='MMClsArchitecture',               # architecture name registered
        model=model),                           # specify defined model to use in the architecture
    distiller=distiller,                        # specify defined distiller to use in the algorithm
    pruner=pruner,                              # specify defined pruner to use in the algorithm
    retraining=False,                           # Bool, specify which stage in the algorithm. True: sunet retrain; False: pre-training supernet
    bn_training_mode=True,                      # set bn to training mode when model is set to eval mode
    input_shape=None)                           # setting input_shape for getting subnet flops

use_ddp_wrapper = True                          # bool, for updating optimizer in train_step to avoid error
```
