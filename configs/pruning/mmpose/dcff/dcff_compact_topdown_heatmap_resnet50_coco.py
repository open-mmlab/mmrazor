_base_ = ['dcff_topdown_heatmap_resnet50_coco.py']

# model settings
model = _base_.model
model['is_deployed'] = True
