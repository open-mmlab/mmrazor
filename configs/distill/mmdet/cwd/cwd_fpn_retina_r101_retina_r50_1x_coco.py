_base_ = ['./cwd_fpn_frcnn_r101_frcnn_r50_1x_coco.py']

model = dict(
    architecture=dict(
        cfg_path='mmdet::retinanet/retinanet_r50_fpn_1x_coco.py',
        pretrained=False),
    teacher=dict(
        cfg_path='mmdet::retinanet/retinanet_r101_fpn_2x_coco.py',
        pretrained=True))
