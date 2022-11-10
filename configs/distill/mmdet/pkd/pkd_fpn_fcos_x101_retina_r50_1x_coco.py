_base_ = ['./pkd_fpn_retina_x101_retina_r50_2x_coco.py']

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/fcos/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco/fcos_x101_64x4d_fpn_gn-head_mstrain_640-800_2x_coco-ede514a8.pth'  # noqa: E501

model = dict(
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_1x_coco.py'),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmdet::fcos/fcos_x101-64x4d_fpn_gn-head_ms-640-800-2x_coco.py'),
    teacher_ckpt=teacher_ckpt)

# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
