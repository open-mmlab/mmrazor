# optimizer
paramwise_cfg = dict(
    bias_decay_mult=0.0, norm_decay_mult=0.0, dwconv_decay_mult=0.0)
optimizer = dict(
    type='SGD',
    lr=0.5,
    momentum=0.9,
    nesterov=True,
    weight_decay=0.0001,
    paramwise_cfg=paramwise_cfg)

optimizer_config = None

# learning policy
lr_config = dict(policy='poly', power=1.0, min_lr=0.0, by_epoch=False)
runner = dict(type='EpochBasedRunner', max_epochs=300)
