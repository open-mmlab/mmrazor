_base_ = './group_fisher_act_prune_resnet50_8xb32_in1k.py'
model = dict(
    mutator=dict(
        channel_unit_cfg=dict(
            default_args=dict(normalization_type='flops', ), ), ), )
