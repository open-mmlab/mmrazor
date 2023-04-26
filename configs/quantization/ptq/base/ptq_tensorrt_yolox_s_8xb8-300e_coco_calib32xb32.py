_base_ = [
    'mmdet::yolox/yolox_s_8xb8-300e_coco.py',
    '../../deploy_cfgs/mmdet/detection_tensorrt-int8-explicit_dynamic-320x320-1344x1344.py'  # noqa: E501
]

_base_.val_dataloader.batch_size = 32

test_cfg = dict(
    type='mmrazor.PTQLoop',
    calibrate_dataloader=_base_.val_dataloader,
    calibrate_steps=32,
)

float_checkpoint = 'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth'  # noqa: E501

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='qint8', bit=8, is_symmetry=True, averaging_constant=0.1),
)

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        pad_size_divisor=32,
        batch_augments=[
            dict(
                type='mmdet.BatchSyncRandomResize',
                random_size_range=(480, 800),
                size_divisor=32,
                interval=10)
        ]),
    architecture=_base_.model,
    deploy_cfg=_base_.deploy_cfg,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.TensorRTQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.predict_by_feat',  # noqa: E501
                'mmdet.models.dense_heads.yolox_head.YOLOXHead.loss_by_feat',
            ])))

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

custom_hooks = []
