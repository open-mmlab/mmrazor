_base_ = [
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

# default_scope = 'mmrazor'
teacher_ckpt = 'faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(
        cfg_path='mmdet::faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py',
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_cwd_fpn0=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn1=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn2=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn3=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10),
            loss_cwd_fpn4=dict(
                type='ChannelWiseDivergence', tau=1, loss_weight=10)),
        loss_forward_mappings=dict(
            loss_cwd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_cwd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_cwd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_cwd_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=3)),
            loss_cwd_fpn4=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=4),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=4)))))

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')
