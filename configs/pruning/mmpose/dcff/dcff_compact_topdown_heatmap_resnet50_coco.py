_base_ = ['dcff_topdown_heatmap_resnet50_coco.py']

# model settings
model = _base_.model
model = _base_.model
# Avoid pruning_ratio check in mutator
model['fix_subnet'] = 'configs/pruning/mmpose/dcff/fix_subnet.yaml'
model['target_pruning_ratio'] = None
model['is_deployed'] = True
