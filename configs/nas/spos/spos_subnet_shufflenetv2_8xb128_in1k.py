_base_ = [
    './spos_supernet_shufflenetv2_8xb128_in1k.py',
]

# FIXME: you may replace this with the mutable_cfg searched by yourself
mutable_cfg = 'https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/nas/spos/spos_shufflenetv2_subnet_8xb128_in1k/spos_shufflenetv2_subnet_8xb128_in1k_flops_0.33M_acc_73.87_20211222-454627be_mutable_cfg.yaml'  # noqa: E501

algorithm = dict(retraining=True, mutable_cfg=mutable_cfg)

runner = dict(max_iters=300000)
find_unused_parameters = False
