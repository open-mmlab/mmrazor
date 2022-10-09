_base_ = [
    'mmdet3d::fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py',
]

train_dataloader = dict(num_workers=4)

student = _base_.model
student.backbone.depth = 50  # using original ResNet50
student.backbone.dcn = None  # no dcn in backbone
student.backbone.stage_with_dcn = (False, False, False, False)
student.backbone.init_cfg.checkpoint = 'open-mmlab://detectron2/resnet50_caffe'

teacher_ckpt = 'https://download.openmmlab.com/mmdetection3d/v0.1.0_models/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'  # noqa: E501
model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='FpnTeacherDistill',
    architecture=student,
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmdet3d::fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py',  # noqa: E501
        pretrained=False),
    teacher_ckpt=teacher_ckpt,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        teacher_recorders=dict(fpn=dict(type='ModuleOutputs', source='neck')),
        distill_losses=dict(
            loss_pkd_fpn0=dict(type='PKDLoss', loss_weight=10),
            loss_pkd_fpn1=dict(type='PKDLoss', loss_weight=10),
            loss_pkd_fpn2=dict(type='PKDLoss', loss_weight=10),
            loss_pkd_fpn3=dict(type='PKDLoss', loss_weight=10)),
        loss_forward_mappings=dict(
            loss_pkd_fpn0=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=0),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=0)),
            loss_pkd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=1),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=1)),
            loss_pkd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=2),
                preds_T=dict(from_student=False, recorder='fpn', data_idx=2)),
            loss_pkd_fpn3=dict(
                preds_S=dict(from_student=True, recorder='fpn', data_idx=3),
                preds_T=dict(from_student=False, recorder='fpn',
                             data_idx=3)))))

find_unused_parameters = True
train_cfg = dict(val_interval=12)
