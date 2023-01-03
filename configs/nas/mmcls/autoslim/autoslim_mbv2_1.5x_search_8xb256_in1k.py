_base_ = ['./autoslim_mbv2_1.5x_supernet_8xb256_in1k.py']

model = dict(bn_training_mode=True)

train_cfg = None
optim_wrapper = None
param_scheduler = None
train_dataloader = None

val_cfg = None
val_dataloader = None
val_evaluator = None

test_cfg = dict(
    _delete_=True,
    type='mmrazor.AutoSlimGreedySearchLoop',
    dataloader=_base_.test_dataloader,
    evaluator=_base_.test_evaluator,
    target_flops=(500., 300., 200.))
