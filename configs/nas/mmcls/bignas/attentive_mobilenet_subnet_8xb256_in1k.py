_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

model = dict(fix_subnet='configs/nas/mmcls/bignas/ATTENTIVE_SUBNET_A6.yaml')

test_cfg = dict(evaluate_fixed_subnet=True)
