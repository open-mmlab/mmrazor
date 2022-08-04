from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

@MODELS.register_module()
class PTQ(BaseAlgorithm):

    observers_enabled: True
    fake_quants_enabled: True

    def __init__(self,
                 architecture,
                 quantizer,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(architecture, data_preprocessor, init_cfg)
        self.quantizer = MODELS.build(quantizer)

    def prepare(self):
        return self.quantizer.prepare(self.architecture)

    def convert(self, model):
        return self.quantizer.convert(model)

    @property
    def state(self):
        return (self.observers_enabled, fake_quants_enabled)

    @state.setter
    def state(self, observers_enabled, fake_quants_enabled):
        for name, submodule in self.architecture.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantizeBase):
                if observers_enabled:
                    submodule.enable_observer()
                else:
                    submodule.disable_observer()
    
                if fake_quants_enabled:
                    submodule.enable_fake_quant()
                else:
                    submodule.disable_fake_quant()
        


