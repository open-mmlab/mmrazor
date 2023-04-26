_base_ = ['./detnas_frcnn_shufflenet_supernet_coco_1x.py']

model = dict(
    _scope_='mmrazor',
    type='sub_model',
    cfg=_base_.supernet,
    # NOTE: You can replace the yaml with the mutable_cfg searched by yourself
    fix_subnet='configs/nas/mmdet/detnas/DETNAS_SUBNET.yaml',
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmrazor/v1/detnas/detnas_subnet_frcnn_shufflenetv2_fpn_1x_coco_bbox_backbone_flops-0.34M_mAP-37.5_20220715-61d2e900_v1.pth',  # noqa: E501
        prefix='architecture.'))

find_unused_parameters = False
