_base_ = [
    '../spos/spos_subnet_shufflenetv2_8xb128_in1k.py',
]

# FIXME: you may replace this with the mutable_cfg searched by yourself
mutable_cfg = 'https://download.openmmlab.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_mutable_cfg.yaml'  # noqa: E501

algorithm = dict(mutable_cfg=mutable_cfg)
