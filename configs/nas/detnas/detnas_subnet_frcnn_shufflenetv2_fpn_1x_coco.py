_base_ = ['./detnas_supernet_frcnn_shufflenetv2_fpn_1x_coco.py']

# FIXME: you may replace this with the mutable_cfg searched by yourself
mutable_cfg = 'https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/nas/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20211222-67fea61f_mutable_cfg.yaml'  # noqa: E501

algorithm = dict(retraining=True, mutable_cfg=mutable_cfg)
