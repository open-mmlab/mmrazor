_base_ = ['./dsnas_supernet_8xb128_in1k.py']

# NOTE: Replace this with the mutable_cfg searched by yourself.
fix_subnet = 'configs/nas/mmcls/dsnas/DSNAS_SUBNET_IMAGENET_PAPER_ALIAS.yaml'

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
