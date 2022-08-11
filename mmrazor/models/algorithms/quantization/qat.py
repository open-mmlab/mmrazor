from mmrazor.registry import MODELS
from ..base import BaseAlgorithm

@MODELS.register_module()
class QAT(BaseAlgorithm):

    def __init__(self,
                 architecture,
                 quantizer,
                 data_preprocessor=None,
                 init_cfg=None):
        super().__init__(architecture, data_preprocessor, init_cfg)
        self.quantizer = MODELS.build(quantizer)
        self.observers_enabled = True
        self.fake_quants_enabled = True

    def prepare(self):
        self.architecture = self.quantizer.prepare(self.architecture)

    def convert(self):
        self.architecture = self.quantizer.convert(self.architecture)

    @property
    def state(self):
        return (self.observers_enabled, self.fake_quants_enabled)

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

        self.observers_enabled = observers_enabled
        self.fake_quants_enabled = fake_quants_enabled
        


