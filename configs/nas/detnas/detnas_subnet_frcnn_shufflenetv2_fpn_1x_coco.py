_base_ = ['./detnas_supernet_frcnn_shufflenetv2_fpn_1x_coco.py']

# FIXME: you may replace this with the mutable_cfg searched by yourself
mutable_cfg = 'configs/nas/detnas/DETNAS_FRCNN_SHUFFLENETV2_340M_COCO_MMRAZOR.yaml'  # noqa: E501

algorithm = dict(retraining=True, mutable_cfg=mutable_cfg)
