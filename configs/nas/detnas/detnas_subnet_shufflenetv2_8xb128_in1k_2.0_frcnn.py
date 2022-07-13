_base_ = ['./detnas_supernet_shufflenetv2_coco_1x_2.0_frcnn.py']

# FIXME: you may replace this with the mutable_cfg searched by yourself
fix_subnet = 'configs/nas/detnas/DETNAS_FRCNN_SHUFFLENETV2_340M_COCO_MMRAZOR_2.0.yaml'  # noqa: E501

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
