_base_ = ['mmdet::paa/paa_r101_fpn_1x_coco.py']

teacher_ckpt = 'http://download.openmmlab.com/mmdetection/v2.0/paa/paa_r101_fpn_1x_coco/paa_r101_fpn_1x_coco_20200821-0a1825a4.pth'  # noqa: E501

student = _base_.model

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='SingleTeacherDistill',
    architecture=student,
    teacher=dict(
        cfg_path='mmdet::paa/paa_r101_fpn_1x_coco.py', pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        distill_deliveries=dict(
            assign=dict(
                type='MethodOutputs',
                max_keep_data=10000,
                method_path='mmdet.models.PAAHead.get_targets'),
            reassign=dict(
                type='MethodOutputs',
                max_keep_data=10000,
                method_path='mmdet.models.PAAHead.paa_reassign'),
        )))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
