#############################################################################
"""# You have to fill these args.

_base_ (str): The path to your pruning config file. pruned_path (str): The path
to the checkpoint of the pruned model.
"""

_base_ = './group_fisher_flops_prune_mobilenet-v2_8xb32_in1k.py'
pruned_path = './work_dirs/group_fisher_flops_prune_mobilenet-v2_8xb32_in1k/flops_0.65.pth'  # noqa
##############################################################################

algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)

# restore lr
optim_wrapper = dict(optimizer=dict(lr=_base_._original_lr_))

# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]

# delete ddp
model_wrapper_cfg = None
