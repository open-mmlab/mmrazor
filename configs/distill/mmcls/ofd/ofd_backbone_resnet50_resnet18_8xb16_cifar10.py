_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

model = dict(
    _scope_='mmrazor',
    type='OverhaulFeatureDistillation',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    architecture=dict(
        cfg_path=  # noqa: E251
        'mmrazor::vanilla/mmcls/wide-resnet/wrn16-w2_b16x8_cifar10.py',
        pretrained=False),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmrazor::vanilla/mmcls/wide-resnet/wrn28-w4_b16x8_cifar10.py',
        pretrained=False),
    teacher_ckpt=  # noqa: E251
    'https://download.openmmlab.com/mmrazor/v1/wide_resnet/wrn28_4_b16x8_cifar10_20220831_173536-d6f8725c.pth',  # noqa: E501
    calculate_student_loss=True,
    student_trainable=True,
    distiller=dict(
        type='OFDDistiller',
        student_recorders=dict(
            bb_1=dict(type='ModuleOutputs', source='backbone.layer2.0.bn1'),
            bb_2=dict(type='ModuleOutputs', source='backbone.layer3.0.bn1'),
            bb_3=dict(type='ModuleOutputs', source='backbone.bn1')),
        teacher_recorders=dict(
            bb_1=dict(type='ModuleOutputs', source='backbone.layer2.0.bn1'),
            bb_2=dict(type='ModuleOutputs', source='backbone.layer3.0.bn1'),
            bb_3=dict(type='ModuleOutputs', source='backbone.bn1')),
        distill_losses=dict(
            loss_1=dict(type='OFDLoss', loss_weight=0.25),
            loss_2=dict(type='OFDLoss', loss_weight=0.5),
            loss_3=dict(type='OFDLoss', loss_weight=1.0)),
        connectors=dict(
            loss_1_sfeat=dict(
                type='ConvModuleConncetor',
                in_channel=32,
                out_channel=64,
                norm_cfg=dict(type='BN'),
                act_cfg=None),
            loss_1_tfeat=dict(type='OFDTeacherConnector'),
            loss_2_sfeat=dict(
                type='ConvModuleConncetor',
                in_channel=64,
                out_channel=128,
                norm_cfg=dict(type='BN'),
                act_cfg=None),
            loss_2_tfeat=dict(type='OFDTeacherConnector'),
            loss_3_sfeat=dict(
                type='ConvModuleConncetor',
                in_channel=128,
                out_channel=256,
                norm_cfg=dict(type='BN'),
                act_cfg=None),
            loss_3_tfeat=dict(type='OFDTeacherConnector')),
        loss_forward_mappings=dict(
            loss_1=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_1',
                    connector='loss_1_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='bb_1',
                    connector='loss_1_tfeat'),
            ),
            loss_2=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_2',
                    connector='loss_2_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='bb_2',
                    connector='loss_2_tfeat'),
            ),
            loss_3=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_3',
                    connector='loss_3_sfeat'),
                t_feature=dict(
                    from_student=False,
                    recorder='bb_3',
                    connector='loss_3_tfeat'),
            ),
        )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
