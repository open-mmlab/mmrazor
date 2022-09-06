_base_ = [
    'mmcls::_base_/datasets/cifar10_bs16.py',
    'mmcls::_base_/schedules/cifar10_bs128.py',
    'mmcls::_base_/default_runtime.py'
]

train_cfg = dict(by_epoch=True, max_epochs=20, val_interval=1)

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
        cfg_path='mmcls::resnet/resnet18_8xb16_cifar10.py', pretrained=False),
    teacher=dict(
        cfg_path='mmcls::resnet/resnet50_8xb16_cifar10.py', pretrained=True),
    teacher_ckpt=  # noqa: E251
    'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_b16x8_cifar10_20210528-f54bfad9.pth',  # noqa: E501
    calculate_student_loss=False,
    student_trainable=False,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            bb_s4=dict(type='ModuleOutputs',
                       source='backbone.layer4.1.conv2')),
        teacher_recorders=dict(
            bb_s4=dict(type='ModuleOutputs',
                       source='backbone.layer4.2.conv3')),
        distill_losses=dict(
            loss_s4_pretrain=dict(type='L2Loss', loss_weight=1.0)),
        connectors=dict(
            loss_s4_sfeat=dict(
                type='Translator', in_channel=512, out_channel=1024),
            loss_s4_tfeat=dict(
                type='Paraphraser',
                phase='pretrain',
                in_channel=2048,
                out_channel=1024)),
        loss_forward_mappings=dict(
            loss_s4_pretrain=dict(
                s_feature=dict(
                    # it actually is t_feature
                    from_student=False,
                    recorder='bb_s4'),
                t_feature=dict(
                    from_student=False,
                    recorder='bb_s4',
                    connector='loss_s4_tfeat'),
            ))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
