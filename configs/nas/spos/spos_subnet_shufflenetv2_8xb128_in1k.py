_base_ = [
    './spos_supernet_shufflenetv2_8xb128_in1k.py',
]

algorithm = dict(retraining=True)

runner = dict(max_iters=300000)
find_unused_parameters = False
