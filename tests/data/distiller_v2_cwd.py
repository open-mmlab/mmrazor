norm_cfg = dict(type='BN', requires_grad=True)

# pspnet r18
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

# pspnet r101
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

# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMSegArchitecture',
        model=student),
    distiller=dict(
        type='SingleTeacherDistillerV2',
        teacher=teacher,
        teacher_trainable=False,
        rewriters = [
            dict(
                function='EncoderDecoder._decode_head_forward_train',
                dependent_module='mmseg.models')
        ],
        student_recorder_cfg=dict( 
            outputs=dict(sources=['decode_head.conv_seg']),
            inputs=dict(sources=['decode_head.conv_seg']),
            weights=dict(sources=['decode_head.conv_seg.weight']),
            functions=dict(
                sources=['EncoderDecoder._decode_head_forward_train'],
                mapping_modules=['mmseg.models'])),
        teacher_recorder_cfg=dict( 
            outputs=dict(sources=['decode_head.conv_seg']),
            inputs=dict(sources=['decode_head.conv_seg']),
            weights=dict(sources=['decode_head.conv_seg.weight']),
            functions=dict(
                sources=['EncoderDecoder._decode_head_forward_train'],
                mapping_modules=['mmseg.models'])),
        components=[
            dict(
                student_items=[
                    dict(source_type='outputs', source='decode_head.conv_seg'),
                    # dict(source_type='inputs', source='decode_head.conv_seg')
                ],
                teacher_items=[
                    dict(source_type='outputs', source='decode_head.conv_seg'),
                    # dict(source_type='inputs', source='decode_head.conv_seg')
                ],
                loss=dict(
                        type='ChannelWiseDivergence',
                        tau=1,
                        loss_weight=5,
                    )),
        ]),
)
