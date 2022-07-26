_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

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
    teacher_ckpt='resnet34_8xb32_in1k_20210831-f257d4e6.pth',
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            feat_4=dict(type='ModuleOutputs', source='backbone.layer4.1.relu'),
            feat_3=dict(type='ModuleOutputs', source='backbone.layer3.1.relu'),
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            feat_4=dict(type='ModuleOutputs', source='backbone.layer4.2.relu'),
            feat_3=dict(type='ModuleOutputs', source='backbone.layer3.5.relu'),
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_f4=dict(type='L2Loss', loss_weight=10),
            loss_f3=dict(type='L2Loss', loss_weight=10),
            loss_kl=dict(
                type='KLDivergence', tau=6, loss_weight=10, reduction='mean')),
        student_connectors=dict(
            loss_f4=dict(
                type='ReLUConnector', in_channel=512, out_channel=2048),
            loss_f3=dict(
                type='ReLUConnector', in_channel=256, out_channel=1024)),
        loss_forward_mappings=dict(
            loss_f4=dict(
                s_feature=dict(
                    from_student=True, recorder='feat_4', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='feat_4', record_idx=2)),
            loss_f3=dict(
                s_feature=dict(
                    from_student=True, recorder='feat_3', record_idx=1),
                t_feature=dict(
                    from_student=False, recorder='feat_3', record_idx=2)),
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
