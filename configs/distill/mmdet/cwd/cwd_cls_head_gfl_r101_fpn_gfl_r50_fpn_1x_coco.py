_base_ = ['./cwd_fpn_retina_r101_retina_r50_1x_coco.py']

model = dict(
    architecture=dict(
        cfg_path='mmdet::gfl/gfl_r50_fpn_1x_coco.py', pretrained=False),
    teacher=dict(
        cfg_path='mmdet::gfl/gfl_r101_fpn_mstrain_2x_coco.py',
        pretrained=True))
