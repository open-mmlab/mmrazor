_base_ = ['dcff_faster_rcnn_resnet50_8xb4_coco.py']

# model settings
model = _base_.model
model['is_deployed'] = True
