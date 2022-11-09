_base_ = ['mmcls::resnet/resnet18_8xb16_cifar10.py']

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QATEpochBasedLoop',
    max_epochs=_base_.train_cfg.max_epochs,
)

model = dict(
    _delete_=True,
    _scope_='mmrazor',
    type='GeneralQuant',
    architecture={{_base_.model}},
    quantizer=dict(
        type='TensorRTQuantizer',
        skipped_methods=[
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ],
        qconfig=dict(
            qtype='affine',
            w_observer=dict(type='mmrazor.MinMaxObserver'),
            a_observer=dict(type='mmrazor.EMAMinMaxObserver'),
            w_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
            a_fake_quant=dict(type='mmrazor.LearnableFakeQuantize'),
            w_qscheme=dict(
                bit=2,
                is_symmetry=False,
                is_per_channel=True,
                is_pot_scale=False,
            ),
            a_qscheme=dict(
                bit=4,
                is_symmetry=False,
                is_per_channel=False,
                is_pot_scale=False),
        )))
