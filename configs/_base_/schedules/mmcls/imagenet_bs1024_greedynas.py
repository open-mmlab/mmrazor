# optimizer
optimizer = dict(
    type='SGD',
    lr=0.12,
    momentum=0.9,
    weight_decay=4e-5)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1.0e-05)
runner = dict(type='IterBasedRunner', max_iters=150000)
