from mmrazor.registry import MODELS
from ..base import BaseAlgorithm
import torch

@MODELS.register_module()
class GeneralQuant(BaseAlgorithm):

    def __init__(self,
                 architecture,
                 quantizer,
                 data_preprocessor=None,
                 init_cfg=None):
        if data_preprocessor is None:
            data_preprocessor = {}
        # The build process is in MMEngine, so we need to add scope here.
        data_preprocessor.setdefault('type', 'mmcls.ClsDataPreprocessor')
        
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
    def state(self, state):
        observers_enabled, fake_quants_enabled = state
        for name, submodule in self.architecture.named_modules():
            if isinstance(submodule, torch.quantization.FakeQuantize):
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
        


