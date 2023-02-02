_base_ = [
    'mmdet::_base_/models/retinanet_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py',
    'mmrazor::_base_/nas_backbones/spos_shufflenet_supernet.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

supernet = _base_.model

supernet.backbone = _base_.nas_backbone
supernet.backbone.norm_cfg = norm_cfg
supernet.backbone.out_indices = (0, 1, 2, 3)
supernet.backbone.with_last_layer = False

supernet.neck.norm_cfg = norm_cfg
supernet.neck.in_channels = [64, 160, 320, 640]

model = dict(
    _delete_=True,
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=dict(type='mmrazor.NasMutator'))

find_unused_parameters = True
