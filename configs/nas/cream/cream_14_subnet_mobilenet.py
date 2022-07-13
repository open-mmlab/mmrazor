_base_ = ['./cream_14_supernet_mobilenet.py']

# FIXME: you may replace this with the mutable_cfg searched by yourself
fix_subnet = 'configs/nas/cream/CREAM_14_MOBILENET_IN1k_2.0.yaml'  # noqa: E501

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
