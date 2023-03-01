#############################################################################
"""# You have to fill these args.

_base_(str): The path to your pruning config file.
pruned_path (str): The path to the checkpoint of the pruned model.
finetune_lr (float): The lr rate to finetune. Usually, we directly use the lr
    rate of the pretrain.
"""

_base_ = './group_fisher_flops_prune_retinanet_r50_fpn_1x_coco.py'
pruned_path = 'https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/retinanet/flops/group_fisher_flops_prune_retinanet_r50_fpn_1x_coco.pth'  # noqa
finetune_lr = 0.005
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
optim_wrapper = dict(optimizer=dict(lr=finetune_lr))

# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]

# delete ddp
model_wrapper_cfg = None
