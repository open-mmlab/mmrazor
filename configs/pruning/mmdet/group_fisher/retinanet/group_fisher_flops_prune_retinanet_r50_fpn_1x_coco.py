_base_ = './group_fisher_act_prune_retinanet_r50_fpn_1x_coco.py'
model = dict(
    mutator=dict(
        channel_unit_cfg=dict(
            default_args=dict(normalization_type='flops', ), ), ), )
