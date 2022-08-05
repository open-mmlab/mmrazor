DefalutQconfigs = dict(
    default = Default,
    tensorrt = TensorRT
)

SupportQtypes = ('affine')
CheckArgs = ['qtype', 'w_qscheme', 'a_qscheme', 'w_fake_quant', 'a_fake_quant', 'w_observer', 'a_observer']

Default = dict(qtype='affine',     # noqa: E241
                w_qscheme=dict(is_symmetry=True, is_per_channel=True, is_pot_scale=False, bit=8, symmetric_range=True),
                a_qscheme=dict(is_symmetry=True, is_per_channel=False, is_pot_scale=False, bit=8, symmetric_range=True),
                w_fake_quant=dict(type='LearnableFakeQuantize'),
                a_fake_quant=dict(type='LearnableFakeQuantize'),
                w_observer=dict(type='MinMaxObserver'),
                a_observer=dict(type='EMAMinMaxObserver'))

TensorRT = dict(qtype='affine',     # noqa: E241
                w_qscheme=dict(is_symmetry=True, is_per_channel=True, is_pot_scale=False, bit=8, symmetric_range=True),
                a_qscheme=dict(is_symmetry=True, is_per_channel=False, is_pot_scale=False, bit=8, symmetric_range=True),
                w_fake_quant=dict(type='LearnableFakeQuantize'),
                a_fake_quant=dict(type='LearnableFakeQuantize'),
                w_observer=dict(type='MinMaxObserver'),
                a_observer=dict(type='EMAMinMaxObserver'))