_base_ = 'autoslim_mbv2_1.5x_slimmable_subnet_8xb256_in1k.py'

_channel_cfg_paths = 'https://download.openmmlab.com/mmrazor/v1/autoslim/autoslim_mbv2_subnet_8xb256_in1k_flops-220M_acc-71.4_20220715-9c288f3b_subnet_cfg.yaml'  # noqa: E501
model = dict(channel_cfg_paths=_channel_cfg_paths)
