_base_ = ['./autoslim_mbv2_1.5x_supernet_8xb256_in1k.py']

train_cfg = dict(
    _delete_=True,
    type='mmrazor.GreedySearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    target_flops=(500., 300., 200.))

val_cfg = dict(_delete_=True)
