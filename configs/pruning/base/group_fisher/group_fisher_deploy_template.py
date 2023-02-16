#############################################################################
"""You have to fill these args.

_base_(str): The path to your pretrain config file.
mutable_cfg (Union[dict,str]): The dict store the pruning structure or the
    json file including it.
divisor (int): The divisor the make the channel number divisible.
"""

_base_ = ''
mutable_cfg = {}
divisor = 8
##############################################################################

architecture = _base_.model

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GroupFisherDeploySubModel',
    architecture=architecture,
    mutable_cfg=mutable_cfg,
    divisor=divisor,
)
