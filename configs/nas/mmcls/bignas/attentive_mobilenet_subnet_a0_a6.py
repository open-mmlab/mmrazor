_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

test_cfg = dict(evaluate_fixed_subnet=False)
