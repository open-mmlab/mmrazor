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
            qtype='affine',
            w_observer=dict(type='mmrazor.MSEObserver'),
            a_observer=dict(type='mmrazor.EMAMSEObserver'),
            w_fake_quant=dict(type='mmrazor.FakeQuantize'),
            a_fake_quant=dict(type='mmrazor.FakeQuantize'),
            w_qscheme=dict(
                bit=8,
                is_symmetry=True,
                is_per_channel=True,
                is_pot_scale=False,
            ),
            a_qscheme=dict(
                bit=8,
                is_symmetry=False,
                is_per_channel=False,
                is_pot_scale=False),
        )))
