_base_ = ['mmcls::resnet/resnet18_8xb16_cifar10.py']

train_cfg = dict(
    _delete_=True,
    type='mmrazor.QATEpochBasedLoop',
    max_epochs=3,
    calibrate_dataloader=None)

model = dict(
    _delete_=True,
    type='mmrazor.QAT',
    architecture=_base_.model,
    quantizer=dict(
        type='mmrazor.CustomQuantizer',
        is_qat=True,
        qconfig=dict(
            qtype='affine',
            w_observer=dict(type='mmrazor.MinMaxObserver'),
            a_observer=dict(type='mmrazor.MinMaxObserver'),
            w_fake_quant=dict(type='mmrazor.BaseFakeQuantize'),
            a_fake_quant=dict(type='mmrazor.BaseFakeQuantize'),
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
        ),
        prepare_custom_config_dict=None,
        convert_custom_config_dict=None,
    ))

# model = dict(
#     type='QAT',
#     architecture=_base_.resnet,
#     quantizer=dict(
#         type='TensorRTQuantizer',
#         example_inputs=(1, 3, 224, 224),
#         is_qat=True)
# )
