_base_ = [
    'mmcls::_base_/datasets/cifar100_bs16_auto_aug.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

model = dict(
    _scope_='mmrazor',
    type='SelfDistill',
    data_preprocessor=dict(
        type='ImgDataPreprocessor',
        # RGB format normalization parameters
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        # convert image from BGR to RGB
        bgr_to_rgb=True),
    architecture=dict(
        cfg_path='mmcls::resnet/resnet18_8xb32_in1k.py', pretrained=False),
    distiller=dict(
        student_recorders=dict(
            bb_s1=dict(type='ModuleOutputs', source='backbone.layer1.1.relu'),
            bb_s2=dict(type='ModuleOutputs', source='backbone.layer2.1.relu'),
            bb_s3=dict(type='ModuleOutputs', source='backbone.layer3.1.relu'),
            data_samples=dict(type='ModuleInputs', source='')),
        distill_losses=dict(
            loss_fet_1=dict(type='L2Loss', normalize=False, mult=0.03),
            loss_label_1=dict(type='CrossEntropyLoss', loss_weight=0.7),
            loss_softl_1=dict(type='KDSoftCELoss', tau=3, loss_weight=0.3),
            loss_fet_2=dict(type='L2Loss', normalize=False, mult=0.03),
            loss_label_2=dict(type='CrossEntropyLoss', loss_weight=0.7),
            loss_softl_2=dict(type='KDSoftCELoss', tau=3, loss_weight=0.3),
            loss_fet_3=dict(type='L2Loss', normalize=False, mult=0.),
            loss_label_3=dict(type='CrossEntropyLoss', loss_weight=0.7),
            loss_softl_3=dict(type='KDSoftCELoss', tau=3, loss_weight=0.3)),
        connectors=dict(
            loss_s1_sfeat=dict(
                type='BYOTConncetor',
                in_channel=64,
                out_channel=512,
                expansion=1,
                num_classes=100),
            loss_s2_sfeat=dict(
                type='BYOTConncetor',
                in_channel=128,
                out_channel=512,
                expansion=1,
                num_classes=100),
            loss_s3_sfeat=dict(
                type='BYOTConncetor',
                in_channel=256,
                out_channel=512,
                expansion=1,
                num_classes=100)),
        loss_forward_mappings=dict(
            loss_fet=dict(
                s_feature=dict(recorder='fc', from_student=True),
                t_feature=dict(recorder='fc', from_student=True)),
            loss_label=dict(
                cls_score=dict(recorder='fc', from_student=True),
                label=dict(recorder='fc', from_student=True)),
            loss_softlabel=dict(
                preds_S=dict(recorder='fc', from_student=True),
                preds_T=dict(recorder='fc', from_student=True)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
