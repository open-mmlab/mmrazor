_base_ = ['dcff_pointrend_resnet50_8xb2_cityscapes.py']

# model settings
model = _base_.model
model['is_deployed'] = True
