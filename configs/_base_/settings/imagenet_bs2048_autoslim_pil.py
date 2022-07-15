_base_ = 'imagenet_bs2048_autoslim.py'

_RandomResizedCrop_cfg = _base_.train_dataloader.dataset.pipeline[1]
assert _RandomResizedCrop_cfg.type == 'mmcls.RandomResizedCrop'
_RandomResizedCrop_cfg.backend = 'pillow'

_ResizeEdge_cfg = _base_.test_dataloader.dataset.pipeline[1]
assert _ResizeEdge_cfg.type == 'mmcls.ResizeEdge'
_ResizeEdge_cfg.backend = 'pillow'
