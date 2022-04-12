_base_ = [
    './mse_inter_fpn_feat_fcos3d_r101_fpn_fcos3d_r50_fpn_1x_nus.py',
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
#workflow = [('val', 1),('train', 1)]

t_weight='/workspace/changyongshu/projects/checkpoints_pretrained/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth'

# model settings
student = dict(
    type='mmdet.FCOSMono3D',
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        ),
    neck=dict(
        in_channels=[64, 128, 256, 512],
        ),
        )

