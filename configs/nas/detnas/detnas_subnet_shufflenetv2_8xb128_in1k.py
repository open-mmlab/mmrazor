_base_ = [
    '../spos/spos_subnet_shufflenetv2_8xb128_in1k.py',
]

# FIXME: you may replace this with the mutable_cfg searched by yourself
mutable_cfg = 'configs/nas/detnas/DETNAS_FRCNN_SHUFFLENETV2_340M_COCO_MMRAZOR.yaml'  # noqa: E501
algorithm = dict(mutable_cfg=mutable_cfg)
