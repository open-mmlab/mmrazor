_base_ = ['mmcls::resnet/resnet18_8xb16_cifar10.py']

test_cfg = dict(
    type='mmrazor.PTQLoop',
)

model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    architecture=_base_.model,
    quantizer=dict(
        type='mmrazor.OpenvinoQuantizer',
        is_qat=False,
        skipped_methods=[
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ],
        qconfig=dict(
            w_observer=dict(type='mmrazor.MovingAveragePerChannelMinMaxObserver'),
            a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
            w_fake_quant=dict(type='mmrazor.FakeQuantize'),
            a_fake_quant=dict(type='mmrazor.FakeQuantize'),
            w_qscheme=dict(
                dtype='uint',
                bit=8,
                is_symmetry=True),
            a_qscheme=dict(
                dtype='uint',
                bit=8,
                is_symmetry=False),
        )))
