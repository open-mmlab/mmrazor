_base_ = ['mmcls::resnet/resnet18_8xb16_cifar10.py']

test_cfg = dict(
    type='mmrazor.PTQLoop',
)

qconfig_1 = dict(
    w_observer=dict(type='mmrazor.MovingAveragePerChannelMinMaxObserver'),
    a_observer=dict(type='mmrazor.MovingAverageMinMaxObserver'),
    w_fake_quant=dict(type='mmrazor.FakeQuantize'),
    a_fake_quant=dict(type='mmrazor.FakeQuantize'),
    w_qscheme=dict(
        bit=8,
        is_symmetry=True),
    a_qscheme=dict(
        bit=8,
        is_symmetry=False),
)

qconfig_mapping = dict(
    global=qconfig_1,
    object_type=[
        ('torch.nn.Conv1d', qconfig_1),
        ('torch.nn.Conv2d', qconfig_1)
    ],
    module_name=[
        ('module1', qconfig_1),
        ('module2', qconfig_1)
    ],
    module_name_regex=[
        ('foo.*', qconfig_1)
    ],
    module_name_object_type_order=[
        ('foo.bar', 'torch.nn.functional.linear', 0, qconfig3)
    ]
)


model = dict(
    _delete_=True,
    type='mmrazor.MMArchitectureQuant',
    architecture=_base_.model,
    quantizer=dict(
        type='mmrazor.OpenvinoQuantizer',
        is_qat=False,
        tracer=dict(
            type='mmrazor.MMQuantizationTracer',
            skipped_module_methods=[
                'mmcls.models.heads.ClsHead._get_loss',
                'mmcls.models.heads.ClsHead._get_predictions'
            ],
            skipped_module_classes=[
                'mmcls.models.heads.ClsHead',
            ],
            skipped_module_names=[
                'module1',
                'module2'
            ]
        )
        qconfig_mapping=dict(
            global=qconfig_1
        )
    )
)
