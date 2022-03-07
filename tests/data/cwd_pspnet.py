# Copyright (c) OpenMMLab. All rights reserved.
norm_cfg = dict(type='BN',requires_grad=True)


student = dict(
    type = 'mmseg.EncoderDecoder',
    backbone=dict(
        type='mmseg.ResNetV1c',
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
        type='mmseg.PSPHead',
        in_channels=512,
        in_index=3,
        channels=128,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        type='mmseg.FCNHead',
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
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)


teacher = dict(
    type = 'mmseg.EncoderDecoder',
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint='pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'),
    backbone=dict(
        type='mmseg.ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True
    ),
    
    decode_head=dict(
        type='mmseg.PSPHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        pool_scales=(1, 2, 3, 6),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
)

algorithm = dict(
    type = 'GeneralDistill',
    architecture = dict(
        type = 'MMSegArchitecture',
        model = student,
    ),
    distiller = dict(
        type = 'SingleTeacherDistiller',
        teacher = teacher,
        teacher_trainable = False,
        components = [
            dict(
                student_module = 'decode_head.conv_seg',
                teacher_module = 'decode_head.conv_seg',
                losses = [
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_logits',
                        tau = 5,
                        loss_weight=3,
                    )
                ]
            )
        ]
    ),
    
)