_base_ = [
    './imagenet_bs1024_spos.py',
]

_RandomResizedCrop_cfg = _base_.train_dataloader.dataset.pipeline[1]
assert _RandomResizedCrop_cfg.type == 'RandomResizedCrop'
_RandomResizedCrop_cfg.crop_ratio_range = (0.25, 1.0)

optim_wrapper = dict(optimizer=dict(weight_decay=1e-4, nesterov=True))

train_dataloader = dict(batch_size=256)

val_dataloader = dict(batch_size=256)

test_dataloader = dict(batch_size=256)
