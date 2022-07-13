_base_ = [
    'mmdet::_base_/models/faster_rcnn_r50_fpn.py',
    'mmdet::_base_/datasets/coco_detection.py',
    'mmdet::_base_/schedules/schedule_1x.py',
    'mmdet::_base_/default_runtime.py'
]

data_root = '/mnt/lustre/share_data/zhangwenwei/data/coco/'

_base_.train_dataloader.dataset.data_root = data_root

visualizer = None

log_level = 'INFO'
load_from = '/mnt/lustre/dongpeijie/detnas_subnet_shufflenetv2_8xb128_in1k_acc-74.08_20211223-92e9b66a_2.0.pth'  # noqa: E501
resume = False

norm_cfg = dict(type='SyncBN', requires_grad=True)
# model settings
_STAGE_MUTABLE = dict(
    _scope_='mmrazor',
    type='mmrazor.OneShotMutableOP',
    candidates=dict(
        shuffle_3x3=dict(
            type='mmrazor.ShuffleBlock', kernel_size=3, norm_cfg=norm_cfg),
        shuffle_5x5=dict(
            type='mmrazor.ShuffleBlock', kernel_size=5, norm_cfg=norm_cfg),
        shuffle_7x7=dict(
            type='mmrazor.ShuffleBlock', kernel_size=7, norm_cfg=norm_cfg),
        shuffle_xception=dict(
            type='mmrazor.ShuffleXception', norm_cfg=norm_cfg),
    ))

arch_setting = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [64, 4, _STAGE_MUTABLE],
    [160, 4, _STAGE_MUTABLE],
    [320, 8, _STAGE_MUTABLE],
    [640, 4, _STAGE_MUTABLE],
]

supernet = _base_.model

supernet.backbone = dict(
    type='mmrazor.SearchableShuffleNetV2',
    arch_setting=arch_setting,
    norm_cfg=norm_cfg,
    out_indices=(0, 1, 2, 3),
    widen_factor=1.0,
    with_last_layer=False)

supernet.neck = dict(
    type='FPN',
    norm_cfg=norm_cfg,
    in_channels=[64, 160, 320, 640],
    out_channels=256,
    num_outs=5)

supernet.roi_head.bbox_head = dict(
    type='Shared4Conv1FCBBoxHead',
    norm_cfg=norm_cfg,
    in_channels=256,
    fc_out_channels=1024,
    roi_feat_size=7,
    num_classes=80,
    bbox_coder=dict(
        type='DeltaXYWHBBoxCoder',
        target_means=[0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2]),
    reg_class_agnostic=False,
    loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    loss_bbox=dict(type='L1Loss', loss_weight=1.0))

mutator = dict(type='mmrazor.OneShotModuleMutator')

fix_subnet = 'configs/nas/detnas/DETNAS_FRCNN_SHUFFLENETV2_340M_COCO_MMRAZOR_2.0.yaml'  # noqa: E501

model = dict(
    _delete_=True,
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=mutator,
    fix_subnet=fix_subnet,
)

find_unused_parameters = True
