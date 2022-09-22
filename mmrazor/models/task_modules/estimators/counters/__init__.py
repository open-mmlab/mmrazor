# Copyright (c) OpenMMLab. All rights reserved.
from .flops_params_counter import get_model_flops_params
from .latency_counter import get_model_latency
from .op_counters import *  # noqa: F401,F403

__all__ = ['get_model_flops_params', 'get_model_latency']
