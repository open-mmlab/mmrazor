_base_ = 'attentive_mobilenet_supernet_8xb256_in1k.py'

model = dict(
    subnet_dict='configs/nas/mmcls/bignas/attentive_mobilenet_subnet.json')
