_base_ = './vgg_group_fisher_prune.py'
custom_imports = dict(imports=['projects'])

algorithm = _base_.model
# `pruned_path` need to be updated.
pruned_path = './work_dirs/vgg_group_fisher_prune_flop/flops_0.30.pth'
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='PruneDeployWrapper',
    algorithm=algorithm,
)

# restore lr
optim_wrapper = dict(optimizer=dict(lr=0.01))
# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]
