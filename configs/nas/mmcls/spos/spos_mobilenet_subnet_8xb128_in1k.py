_base_ = ['./spos_mobilenet_supernet_8xb128_in1k.py']

# FIXME: you may replace this with the mutable_cfg searched by yourself
fix_subnet = 'configs/nas/spos/AngleNAS_SHUFFLENETV2_IN1k_2.0.yaml'  # noqa: E501

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
