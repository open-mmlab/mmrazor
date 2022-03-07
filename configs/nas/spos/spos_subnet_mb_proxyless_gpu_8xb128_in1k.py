_base_ = [
    './spos_supernet_mb_proxyless_gpu_8xb128_in1k.py',
]

algorithm = dict(retraining=True)
evaluation = dict(interval=10000, metric='accuracy')
checkpoint_config = dict(interval=30000)

runner = dict(max_iters=300000)
find_unused_parameters = False
