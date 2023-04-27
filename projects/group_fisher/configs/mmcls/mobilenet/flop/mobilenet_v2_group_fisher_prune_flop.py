_base_ = '../mobilenet_v2_group_fisher_prune.py'
model = dict(
    mutator=dict(
        channel_unit_cfg=dict(default_args=dict(detla_type='flop', ), ), ), )
