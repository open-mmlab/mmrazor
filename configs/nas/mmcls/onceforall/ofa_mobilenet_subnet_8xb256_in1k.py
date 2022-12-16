_base_ = 'ofa_mobilenet_supernet_32xb64_in1k.py'

model = dict(
    fix_subnet='configs/nas/mmcls/onceforall/OFA_SUBNET_NOTE8_LAT22.yaml')

test_cfg = dict(evaluate_fixed_subnet=True)
