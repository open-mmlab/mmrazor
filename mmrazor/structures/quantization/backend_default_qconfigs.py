from .qscheme import QuantizeScheme

DefalutQconfigs = dict(
    default = Default,
    tensorrt = TensorRT
) 

Default = dict(qtype='affine',     # noqa: E241
                w_qscheme=QuantizeScheme(is_symmetry=True, is_per_channel=True, is_pot_scale=False, bit=8, symmetric_range=True),
                a_qscheme=QuantizeScheme(is_symmetry=True, is_per_channel=False, is_pot_scale=False, bit=8, symmetric_range=True),
                w_fake_quant=dict(type=LearnableFakeQuantize),
                a_fake_quant=dict(LearnableFakeQuantize),
                w_observer=dict(MinMaxObserver),
                a_observer=dict(EMAMinMaxObserver))

TensorRT = dict(qtype='affine',     # noqa: E241
                w_qscheme=QuantizeScheme(is_symmetry=True, is_per_channel=True, is_pot_scale=False, bit=8, symmetric_range=True),
                a_qscheme=QuantizeScheme(is_symmetry=True, is_per_channel=False, is_pot_scale=False, bit=8, symmetric_range=True),
                w_fake_quant=dict(type=LearnableFakeQuantize),
                a_fake_quant=dict(LearnableFakeQuantize),
                w_observer=dict(MinMaxObserver),
                a_observer=dict(EMAMinMaxObserver))