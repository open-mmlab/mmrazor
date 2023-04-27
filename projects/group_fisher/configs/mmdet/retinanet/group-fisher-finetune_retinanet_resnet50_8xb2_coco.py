_base_ = './group-fisher-pruning_retinanet_resnet50_8xb2_coco.py'

algorithm = _base_.model
# `pruned_path` need to be updated.
pruned_path = './work_dirs/group-fisher-pruning_retinanet_resnet50_8xb2_coco/flops_0.50.pth'  # noqa
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

# delete ddp
model_wrapper_cfg = None
