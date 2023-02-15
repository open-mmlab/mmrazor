# yapf: disable
# flake8: noqa
#############################################################################
# You have to fill these args.
_base_ = './group_fisher_act_resnet50_8xb32_in1k.py'  # config to prune your model

pruned_path = './work_dirs/group_fisher_act_resnet50_8xb32_in1k/flops_0.50.pth'  # path of the checkpoint of the pruned model.
##############################################################################
# yapf: enable

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
