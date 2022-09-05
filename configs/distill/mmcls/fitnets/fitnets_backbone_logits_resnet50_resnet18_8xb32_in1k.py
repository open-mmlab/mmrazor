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
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.1.relu'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.1.relu'),
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs', source='backbone.layer4.2.relu'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.5.relu'),
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_s4=dict(type='L2Loss', loss_weight=10),
            loss_s3=dict(type='L2Loss', loss_weight=10),
            loss_kl=dict(
                type='KLDivergence', tau=6, loss_weight=10, reduction='mean')),
        connectors=dict(
            loss_s4_sfeat=dict(
                type='ConvModuleConncetor',
                in_channel=512,
                out_channel=2048,
                norm_cfg=dict(type='BN')),
            loss_s3_sfeat=dict(
                type='ConvModuleConncetor',
                in_channel=256,
                out_channel=1024,
                norm_cfg=dict(type='BN'))),
        loss_forward_mappings=dict(
            loss_s4=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s4',
                    record_idx=1,
                    connector='loss_s4_sfeat'),
                t_feature=dict(
                    from_student=False, recorder='bb_s4', record_idx=2)),
            loss_s3=dict(
                s_feature=dict(
                    from_student=True,
                    recorder='bb_s3',
                    record_idx=1,
                    connector='loss_s3_sfeat'),
                t_feature=dict(
                    from_student=False, recorder='bb_s3', record_idx=2)),
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
