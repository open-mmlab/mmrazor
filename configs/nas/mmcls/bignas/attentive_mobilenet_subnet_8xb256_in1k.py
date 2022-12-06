_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

fix_subnet = 'configs/nas/mmcls/bignas/BIGNAS_SUBNET_MAX.yaml'

model = dict(fix_subnet=fix_subnet)

test_cfg = dict(evaluate_fixed_subnet=True)
