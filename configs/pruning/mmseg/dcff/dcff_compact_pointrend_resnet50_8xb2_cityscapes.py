_base_ = ['dcff_pointrend_resnet50_8xb2_cityscapes.py']

# model settings
model = _base_.model
# Avoid pruning_ratio check in mutator
model['fix_subnet'] = 'configs/pruning/mmseg/dcff/fix_subnet.yaml'
model['target_pruning_ratio'] = None
model['is_deployed'] = True
