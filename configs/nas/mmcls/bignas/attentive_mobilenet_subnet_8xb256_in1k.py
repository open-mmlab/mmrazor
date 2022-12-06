_base_ = 'attentive_mobilenet_supernet_32xb64_in1k.py'

fix_subnet = 'configs/nas/mmcls/bignas/BIGNAS_SUPERNET.yaml'

model = dict(fix_subnet=fix_subnet)
