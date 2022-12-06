_base_ = ['dcff_faster_rcnn_resnet50_8xb4_coco.py']

# model settings
model = _base_.model
model = _base_.model
# Avoid pruning_ratio check in mutator
model['fix_subnet'] = 'configs/pruning/mmdet/dcff/fix_subnet.yaml'
model['target_pruning_ratio'] = None
model['is_deployed'] = True
