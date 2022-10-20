from mmengine.hooks import CheckpointHook
from mmengine.hooks import Hook
from mmrazor.registry import HOOKS

@HOOKS.register_module()
class QuantCheckpointHook(CheckpointHook):
    
