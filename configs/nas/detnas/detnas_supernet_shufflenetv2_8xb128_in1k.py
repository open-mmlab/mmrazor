_base_ = [
    '../spos/spos_shufflenetv2_supernet_8xb128_in1k.py',
]

runner = dict(max_iters=300000)
