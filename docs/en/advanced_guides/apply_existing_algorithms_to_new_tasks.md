# Apply existing algorithms to new tasks

Here we show how to apply existing algorithms to other tasks with an example of [SPOS ](https://github.com/open-mmlab/mmrazor/tree/main/configs/nas/mmcls/spos)& [DetNAS](https://github.com/open-mmlab/mmrazor/tree/main/configs/nas/mmdet/detnas).

> SPOS: Single Path One-Shot NAS for classification
>
> DetNAS: Single Path One-Shot NAS for detection

**You just need to configure the existing algorithms in your config only by replacing** **the architecture of** **mmcls** **with** **mmdet** **'s**

You can implement a new algorithm by inheriting from the existing algorithm quickly if the new task's specificity leads to the failure of applying directly.

SPOS config VS DetNAS config

- SPOS

```Python
_base_ = [
    'mmrazor::_base_/settings/imagenet_bs1024_spos.py',
    'mmrazor::_base_/nas_backbones/spos_shufflenet_supernet.py',
    'mmcls::_base_/default_runtime.py',
]

# model
supernet = dict(
    type='ImageClassifier',
    data_preprocessor=_base_.preprocess_cfg,
    backbone=_base_.nas_backbone,
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=1024,
        loss=dict(
            type='LabelSmoothLoss',
            num_classes=1000,
            label_smooth_val=0.1,
            mode='original',
            loss_weight=1.0),
        topk=(1, 5)))

model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=dict(type='mmrazor.OneShotModuleMutator'))

find_unused_parameters = True
```

- DetNAS

```Python
_base_ = [
    'mmdet::_base_/models/faster-rcnn_r50_fpn.py',
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

supernet.roi_head.bbox_head.norm_cfg = norm_cfg
supernet.roi_head.bbox_head.type = 'Shared4Conv1FCBBoxHead'

model = dict(
    _delete_=True,
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=dict(type='mmrazor.OneShotModuleMutator'))

find_unused_parameters = True
```
