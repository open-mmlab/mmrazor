_base_ = ['mmcls::resnet/resnet50_8xb32_in1k.py']

train_dataloader=dict(
    batch_size=32
)

test_cfg = dict(
    type='mmrazor.PTQLoop',
    calibrate_dataloader=train_dataloader,
    calibrate_steps=32,
)

global_qconfig=dict(
    w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        qdtype='qint8',
        bit=8,
        is_symmetry=True,
        is_symmetric_range=True),
    a_qscheme=dict(
        qdtype='quint8',
        bit=8,
        is_symmetry=True,
        averaging_constant=0.1),
)

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    architecture=_base_.model,
    float_checkpoint='/tmp/humu/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
    quantizer=dict(
        type='mmrazor.OpenVINOQuantizer',
        global_qconfig=global_qconfig,
        # no_observer_modules=[
        #     'mmcv.cnn.bricks.DropPath'
        # ],
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ]
        )
    )
)
