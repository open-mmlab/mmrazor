# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import Config
from mmengine.registry import MODELS
from mmrazor.models.utils import CustomTracer

cfg_path = 'configs/quantization/ptq/demo.py'

def main():
    # load config
    cfg = Config.fromfile(cfg_path)
    model = MODELS.build(cfg.model)
    tracer = CustomTracer()
    graph = tracer.trace(model)
    print(graph)
    

if __name__ == '__main__':
    main()