_base_ = [
    './spos_supernet_shufflenetv2_8xb128_in1k.py',
]

# FIXME: you may replace this with the mutable_cfg searched by yourself
mutable_cfg = 'configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml'

algorithm = dict(retraining=True, mutable_cfg=mutable_cfg)

runner = dict(max_iters=300000)
find_unused_parameters = False
