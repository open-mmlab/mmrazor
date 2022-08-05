_base_ = [
    'mmcls::_base_/datasets/imagenet_bs32.py',
    'mmcls::resnet/resnet34_8xb32_in1k.py',
    'mmcls::_base_/schedules/imagenet_bs256.py',
    'mmcls::_base_/default_runtime.py'
]

train_cfg = dict(
    type='QATEpochBasedLoop',
    max_epochs=20,
    calibrate_dataloader=None
)

# model = dict(
#     type='QAT',
#     architecture=_base_.resnet,
#     quantizer=dict(
#         type='TensorRTQuantizer',
#         example_inputs=(1, 3, 224, 224),
#         is_qat=True)
# )

model = dict(
    type='QAT',
    architecture=_base_.resnet,
    quantizer=dict(
        type='BaseQuantizer',
        example_inputs=(1, 3, 224, 224),
        is_qat=True,
        qconfig=dict(
            qtype='affine',
            w_observer=dict(type='MSEObserver'),
            a_observer=dict(type='EMAMSEObserver'),
            w_fakequantize=dict(type='AdaRoundFakeQuantize'),
            a_fakequantize=dict(type='FixedFakeQuantize')，
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
        prepare_custom_config=None,
        convert_custom_config=None,
    )
)