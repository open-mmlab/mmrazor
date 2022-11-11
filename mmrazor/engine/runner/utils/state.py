# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.logging import print_log
from torch.ao.quantization import FakeQuantize


# TODO: may be removed
def set_quant_state(model, enable_observer=True, enable_fake_quant=True):
    for name, submodule in model.named_modules():
        if isinstance(submodule, FakeQuantize):
            if enable_observer:
                submodule.enable_observer()
            else:
                submodule.disable_observer()
            if enable_fake_quant:
                submodule.enable_fake_quant()
            else:
                submodule.disable_fake_quant()
    print_log(f'Enable observer: {enable_observer}; \
                Enable fake quant: {enable_fake_quant}')
