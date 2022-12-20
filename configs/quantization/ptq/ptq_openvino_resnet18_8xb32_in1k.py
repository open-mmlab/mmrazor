_base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py']

test_cfg = dict(
    type='mmrazor.PTQLoop'
)

qconfig_mapping = dict(
    _global_=dict(
        w_observer=dict(type='mmrazor.PerChannelMinMaxObserver'),
        a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
        w_fake_quant=dict(type='mmrazor.FakeQuantize'),
        a_fake_quant=dict(type='mmrazor.FakeQuantize'),
        w_qscheme=dict(
            qdtype='quint8',
            bit=8,
            is_symmetry=False),
        a_qscheme=dict(
            qdtype='quint8',
            bit=8,
            is_symmetry=False),
    )
)

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    architecture=_base_.model,
    float_checkpoint='/tmp/humu/resnet18_8xb32_in1k_20210831-fbbb1da6.pth',
    quantizer=dict(
        type='mmrazor.WithoutDeployQuantizer',
        qconfig_mapping=qconfig_mapping,
        tracer=dict(
            type='mmrazor.CustomTracer',
            skipped_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ]
        )
    )
)
