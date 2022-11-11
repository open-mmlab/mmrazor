_base_ = ['./spos_shufflenet_supernet_8xb128_in1k.py']

# FIXME: you may replace this with the searched by yourself
fix_subnet = 'configs/nas/mmcls/spos/SPOS_SUBNET.yaml'

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
