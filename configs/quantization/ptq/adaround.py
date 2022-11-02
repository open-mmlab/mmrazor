_base_ = ['mmcls::resnet/resnet18_8xb32_in1k.py']

test_cfg = dict(
    _delete_=True,
    type='mmrazor.PTQLoop',
    dataloader=_base_.test_dataloader,
    evaluator=_base_.test_evaluator,
    calibrate_dataloader=_base_.train_dataloader,
    batch_num=32,
    # reconstruction_cfg=dict(
    #     pattern='layer',
    #     loss=dict(
    #         type='mmrazor.AdaRoundLoss',
    #         iters=20000
    #     )
    # )
)

model = dict(
    _delete_=True,
    type='mmrazor.GeneralQuant',
    architecture=_base_.model,
    quantizer=dict(
        type='mmrazor.CustomQuantizer',
        is_qat=False,
        skipped_methods=[
            'mmcls.models.heads.ClsHead._get_loss',
            'mmcls.models.heads.ClsHead._get_predictions'
        ],
        qconfig=dict(
            qtype='affine',
            w_observer=dict(type='mmrazor.MSEObserver'),
            a_observer=dict(type='mmrazor.EMAMSEObserver'),
            w_fake_quant=dict(type='mmrazor.AdaRoundFakeQuantize'),
            a_fake_quant=dict(type='mmrazor.FakeQuantize'),
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
