_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

teacher_ckpt = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth'  # noqa: E501

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
        cfg_path='mmcls::resnet/resnet34_8xb32_in1k.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(
                type='DISTLoss',
                inter_loss_weight=1.0,
                intra_loss_weight=1.0,
                tau=1,
                loss_weight=2,
            )),
        loss_forward_mappings=dict(
            loss_kl=dict(
                logits_S=dict(from_student=True, recorder='fc'),
                logits_T=dict(from_student=False, recorder='fc')))))

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

optim_wrapper = dict(optimizer=dict(nesterov=True))
