_base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py']

train_dataloader = dict(batch_size=1024)

global_qconfig = dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8',
        bit=8,
        is_symmetry=True,
        is_symmetric_range=True,
    ),
    a_qscheme=dict(
        qdtype='quint8',
        bit=8,
        is_symmetry=True,
        averaging_constant=0.1,
    ),
)

float_checkpoint = 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth'  # noqa: E501

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    architecture=_base_.model,
    float_checkpoint=float_checkpoint,
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ])))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.004, momentum=0.9, weight_decay=0.0001))

# learning policy
param_scheduler = dict(
    _delete_=True,
    type='CosineAnnealingLR',
    T_max=100,
    by_epoch=True,
    begin=0,
    end=100)

model_wrapper_cfg = dict(
    type='mmrazor.MMArchitectureQuantDDP',
    broadcast_buffers=False,
    find_unused_parameters=True)

# train, val, test setting
train_cfg = dict(
    _delete_=True,
    type='mmrazor.QATEpochBasedLoop',
    max_epochs=100,
    val_interval=1)
val_cfg = dict(_delete_=True, type='mmrazor.QATValLoop')
