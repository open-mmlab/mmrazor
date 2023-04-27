_base_ = './group-fisher-pruning_retinanet_resnet50_8xb2_coco.py'

model = dict(
    mutator=dict(
        channel_unit_cfg=dict(default_args=dict(detla_type='act', ), ), ), )
