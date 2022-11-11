_base_ = ['./detnas_frcnn_shufflenet_supernet_coco_1x.py']

# FIXME: you may replace this with the searched by yourself
fix_subnet = 'configs/nas/mmdet/detnas/DETNAS_SUBNET.yaml'

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
