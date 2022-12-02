from abc import abstractmethod
import torch
from mmrazor.registry import MODELS
from mmengine.model import BaseModule

@MODELS.register_module()
class BaseQuantizer(BaseModule):
    def __init__(self,
                 tracer,
                 init_cfg=None):
        super().__init__(init_cfg):
    
    @abstractmethod
    def prepare(self):
        pass
    
    def build_tracer(self):
        pass


@MODELS.register_module()
class WithoutBackendQuantizer(BaseModule):
    def __init__(self,
                 qconfig_mapping,
                 tracer=dict(type='MMTracer'),
                 init_cfg=None):
        super().__init__(tracer, init_cfg):
    
    def prepare(self, model, graph_module):
        pass

    def build_qconfig_mapping(self, qconfig_mapping):
        pass

@MODELS.register_module()
class WithBackendQuantizer(BaseModule):
    def __init__(self,
                 bits,
                 observers,
                 fakequants,
                 backend_config,
                 tracer=dict(type='MMTracer'),
                 init_cfg=None):
        super().__init__(tracer, init_cfg):
    
    def prepare(self, model, graph_module):
        pass
    
    def build_backend_config(self, backend_config):
        pass
    
    def post_process_weight_fakequant(self):
        pass

    def prepare_for_mmdeploy(self, model):
        pass
    

