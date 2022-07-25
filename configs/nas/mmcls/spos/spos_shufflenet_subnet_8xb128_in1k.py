_base_ = ['./spos_shufflenet_supernet_8xb128_in1k.py']

# FIXME: you may replace this with the mutable_cfg searched by yourself
fix_subnet = 'https://download.openmmlab.com/mmrazor/v1/spos/spos_shufflenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20220715-aa94d5ef_subnet_cfg_v1.yaml'  # noqa: E501

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
