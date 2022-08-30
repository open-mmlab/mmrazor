_base_ = ['./cwd_fpn_retina_r101_retina_r50_1x_coco.py']

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/gfl/gfl_r101_fpn_mstrain_2x_coco/gfl_r101_fpn_mstrain_2x_coco_20200629_200126-dd12f847.pth'  # noqa: E501
model = dict(
    architecture=dict(
        cfg_path='mmdet::gfl/gfl_r50_fpn_1x_coco.py', pretrained=False),
    teacher=dict(
        cfg_path='mmdet::gfl/gfl_r101_fpn_ms-2x_coco.py', pretrained=True),
    teacher_ckpt=teacher_ckpt)
