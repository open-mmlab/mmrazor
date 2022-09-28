_base_ = [
    'mmdet3d::pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
]

student = _base_.model
teacher = _base_.model

teacher_ckpt = 's3://caoweihan/pretrained/det3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='FpnTeacherDistill',
    architecture=student,
    teacher=teacher,
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fpn=dict(type='ModuleOutputs', source='pts_neck')),
        teacher_recorders=dict(
            fpn=dict(type='ModuleOutputs', source='pts_neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=5),
            loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=5),
            loss_pkd_fpn2=dict(type='PKDLoss', loss_weight=5)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=2)))))

train_dataloader = dict(num_workers=4)
find_unused_parameters = True
