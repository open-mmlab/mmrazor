#############################################################################
"""# You have to fill these args.

_base_(str): The path to your pruning config file.
pruned_path (str): The path to the checkpoint of the pruned model.
finetune_lr (float): The lr rate to finetune. Usually, we directly use the lr
    rate of the pretrain.
"""

_base_ = './group_fisher_act_prune_resnet50_8xb32_in1k.py'
pruned_path = 'https://download.openmmlab.com/mmrazor/v1/pruning/group_fisher/resnet50/act/group_fisher_act_prune_resnet50_8xb32_in1k.pth'  # noqa
finetune_lr = 0.1
##############################################################################

algorithm = _base_.model
algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

teacher = algorithm.architecture

pruned = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherSubModel',
    algorithm=algorithm,
)

model = dict(
    _scope_='mmrazor',
    _delete_=True,
    type='SingleTeacherDistill',
    data_preprocessor=None,
    architecture=pruned,
    teacher=teacher,
    distiller=dict(
        type='ConfigurableDistiller',
        student_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        teacher_recorders=dict(
            fc=dict(type='ModuleOutputs', source='head.fc')),
        distill_losses=dict(
            loss_kl=dict(type='DISTLoss', tau=1, loss_weight=1)),
        loss_forward_mappings=dict(
            loss_kl=dict(
                preds_S=dict(from_student=True, recorder='fc'),
                preds_T=dict(from_student=False, recorder='fc')))))

find_unused_parameters = True
val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')

# restore lr
optim_wrapper = dict(optimizer=dict(lr=finetune_lr))

# remove pruning related hooks
custom_hooks = _base_.custom_hooks[:-2]

# delete ddp
model_wrapper_cfg = None

_base_ = './resnet_group_fisher_prune.py'

# 76.3520
