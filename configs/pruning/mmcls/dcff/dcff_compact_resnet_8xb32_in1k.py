_base_ = ['dcff_resnet_8xb32_in1k.py']

# model settings
model = _base_.model
model['is_deployed'] = True
