_base_ = ['./detnas_frcnn_shufflenet_supernet_coco_1x.py']

# FIXME: you may replace this with the mutable_cfg searched by yourself
# fix_subnet = 'configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER_2.0.yaml'  # noqa: E501
fix_subnet = 'configs/nas/detnas/DetNAS_SPOS_SHUFFLENETV2_330M_IN1k_PAPER_2.0.yaml'  # noqa: E501

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
