# optimizer
optimizer = dict(
    type='RMSpropTF', lr=0.064, eps=0.001, weight_decay=1e-5, momentum=0.9)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='Step',
    step=3000,
    gamma=0.97,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1250 * 3,
    warmup_ratio=1e-6 / 0.064,
    warmup_by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=450 * 1250)
