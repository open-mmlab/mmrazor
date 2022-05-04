_base_ = [
    './spos_supernet_mobilenet_proxyless_gpu_8xb128_in1k.py',
]

# FIXME: you may replace this with the mutable_cfg searched by yourself
mutable_cfg = 'https://download.openmmlab.com/mmrazor/v0.1/nas/spos/spos_mobilenet_subnet/spos_angelnas_flops_0.49G_acc_75.98_20220307-54f4698f_mutable_cfg.yaml'  # noqa: E501

algorithm = dict(retraining=True, mutable_cfg=mutable_cfg)
evaluation = dict(interval=10000, metric='accuracy')
checkpoint_config = dict(interval=30000)

runner = dict(max_iters=300000)
find_unused_parameters = False
