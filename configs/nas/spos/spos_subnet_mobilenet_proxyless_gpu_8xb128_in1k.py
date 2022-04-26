_base_ = [
    './spos_supernet_mobilenet_proxyless_gpu_8xb128_in1k.py',
]

# FIXME: you may replace this with the mutable_cfg searched by yourself
mutable_cfg = 'configs/nas/spos/SPOS_MOBILENET_490M_FROM_ANGELNAS.yaml'

algorithm = dict(retraining=True, mutable_cfg=mutable_cfg)
evaluation = dict(interval=10000, metric='accuracy')
checkpoint_config = dict(interval=30000)

runner = dict(max_iters=300000)
find_unused_parameters = False
