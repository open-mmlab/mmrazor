_base_ = './mobilenet_v2_group_fisher_prune.py'
custom_imports = dict(imports=['projects'])

algorithm = _base_.model
pruned_path = './work_dirs/mobilenet_v2_group_fisher_prune/flops_0.65.pth'
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='PruneDeployWrapper',
    algorithm=algorithm,
)

# restore optimizer

optim_wrapper = dict(
    _delete_=True,
    optimizer=dict(
        type='SGD',
        lr=0.045,
        momentum=0.9,
        weight_decay=4e-05,
        _scope_='mmcls'))
param_scheduler = dict(
    _delete_=True,
    type='StepLR',
    by_epoch=True,
    step_size=1,
    gamma=0.98,
    _scope_='mmcls')

# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]

# delete ddp
model_wrapper_cfg = None
