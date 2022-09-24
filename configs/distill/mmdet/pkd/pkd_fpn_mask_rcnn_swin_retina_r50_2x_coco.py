_base_ = ['./pkd_fpn_frcnn_r101_frcnn_r50_2x_coco.py']

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth'  # noqa: E501

model = dict(
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_2x_coco.py'),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmdet::swin/mask-rcnn_swin-s-p4-w7_fpn_amp-ms-crop-3x_coco.py'),
    teacher_ckpt=teacher_ckpt)

# optimizer
optim_wrapper = dict(optimizer=dict(lr=0.01))
