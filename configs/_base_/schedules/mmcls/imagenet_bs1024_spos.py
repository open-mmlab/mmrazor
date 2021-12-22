# optimizer
paramwise_cfg = dict(bias_decay_mult=0.0, norm_decay_mult=0.0)
optimizer = dict(
    type='SGD',
    lr=0.5,
    momentum=0.9,
    weight_decay=4e-5,
    paramwise_cfg=paramwise_cfg)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='poly', power=1.0, min_lr=0.0, by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=300000)
