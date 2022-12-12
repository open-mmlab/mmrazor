_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    architecture=dict(
        cfg_path='mmcls::resnet/resnet18_8xb32_in1k.py', pretrained=False),
    teacher=dict(
        cfg_path='mmcls::resnet/resnet50_8xb32_in1k.py', pretrained=True),
    teacher_ckpt=teacher_ckpt,
    calculate_student_loss=False,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.1.conv2'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.1.conv2'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.1.conv2'),
            bb_s1=dict(type='ModuleOutputs',
                       source='backbone.layer1.1.conv2')),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.conv3'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.5.conv3'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.3.conv3'),
            bb_s1=dict(type='ModuleOutputs',
                       source='backbone.layer1.2.conv3')),
        distill_losses=dict(
            loss_s4=dict(type='ABLoss', loss_weight=1.0),
            loss_s3=dict(type='ABLoss', loss_weight=0.5),
            loss_s2=dict(type='ABLoss', loss_weight=0.25),
            loss_s1=dict(type='ABLoss', loss_weight=0.125)),
        connectors=dict(
            loss_s4_sfeat=dict(
                type='ConvModuleConnector',
                in_channel=512,
                out_channel=2048,
                norm_cfg=dict(type='BN'),
                act_cfg=None),
            loss_s3_sfeat=dict(
                type='ConvModuleConnector',
                in_channel=256,
                out_channel=1024,
                norm_cfg=dict(type='BN'),
                act_cfg=None),
            loss_s2_sfeat=dict(
                type='ConvModuleConnector',
                in_channel=128,
                out_channel=512,
                norm_cfg=dict(type='BN'),
                act_cfg=None),
            loss_s1_sfeat=dict(
                type='ConvModuleConnector',
                in_channel=64,
                out_channel=256,
                norm_cfg=dict(type='BN'),
                act_cfg=None)),
        loss_forward_mappings=dict(
            loss_s4=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s4',
                    connector='loss_s4_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s4')),
            loss_s3=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s3',
                    connector='loss_s3_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s3')),
            loss_s2=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s2',
                    connector='loss_s2_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s2')),
            loss_s1=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s1',
                    connector='loss_s1_sfeat'),
                t_feature=dict(from_student=False, recorder='bb_s1')))))

find_unused_parameters = True

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
