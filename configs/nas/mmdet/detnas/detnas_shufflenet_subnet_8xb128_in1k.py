_base_ = './detnas_shufflenet_supernet_8xb128_in1k.py'

# FIXME: you may replace this with the mutable_cfg searched by yourself
fix_subnet = 'https://download.openmmlab.com/mmrazor/v1/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20220715-61d2e900_subnet_cfg_v1.yaml'  # noqa: E501

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
