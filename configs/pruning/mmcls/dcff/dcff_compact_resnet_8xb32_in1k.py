_base_ = ['dcff_resnet_8xb32_in1k.py']

# model settings
model = _base_.model
# Avoid pruning_ratio check in mutator
model['fix_subnet'] = 'configs/pruning/mmcls/dcff/fix_subnet.yaml'
model['target_pruning_ratio'] = None
model['is_deployed'] = True
