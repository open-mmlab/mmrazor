_base_ = [
    '../spos/spos_supernet_shufflenetv2_8xb128_in1k.py',
]

runner = dict(max_iters=300000)
