_base_ = ['./pkd_fpn_retina_x101_retina_r50_2x_coco.py']

teacher_ckpt = 'https://download.openmmlab.com/mmdetection/v2.0/reppoints/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco/reppoints_moment_x101_fpn_dconv_c3-c5_gn-neck%2Bhead_2x_coco_20200329-f87da1ea.pth'  # noqa: E501

model = dict(
    architecture=dict(
        cfg_path=  # noqa: E251
        'mmdet::reppoints/reppoints-moment_r50_fpn-gn_head-gn_2x_coco.py'),
    teacher=dict(
        cfg_path=  # noqa: E251
        'mmdet::reppoints/reppoints-moment_x101-dconv-c3-c5_fpn-gn_head-gn_2x_coco.py'  # noqa: E501
    ),
    teacher_ckpt=teacher_ckpt)
