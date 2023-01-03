_base_ = 'imagenet_bs2048_autoslim.py'

_RandomResizedCrop_cfg = _base_.train_dataloader.dataset.pipeline[1]
assert _RandomResizedCrop_cfg.type == 'RandomResizedCrop'
_RandomResizedCrop_cfg.backend = 'pillow'

_ResizeEdge_cfg_val = _base_.val_dataloader.dataset.pipeline[1]
assert _ResizeEdge_cfg_val.type == 'ResizeEdge'
_ResizeEdge_cfg_val.backend = 'pillow'

_ResizeEdge_cfg_test = _base_.test_dataloader.dataset.pipeline[1]
assert _ResizeEdge_cfg_test.type == 'ResizeEdge'
_ResizeEdge_cfg_test.backend = 'pillow'
