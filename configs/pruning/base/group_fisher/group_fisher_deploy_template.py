# yapf: disable
# flake8: noqa
#############################################################################
# You have to fill these args.
_base_ = 'path to the pretrain config'  # config to prune your model.
mutable_cfg = {} # config of the mutable channel unit.
divisor=8 # the divisor the make the channel number divisible.

##############################################################################
# yapf: enable

architecture = _base_.model
# algorithm.init_cfg = dict(type='Pretrained', checkpoint=pruned_path)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherDeploySubModel',
    architecture=architecture,
    mutable_cfg=mutable_cfg,
)
