# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor import digit_version

if digit_version(torch.__version__) >= digit_version('1.13.0'):
    from .backend_config import *  # noqa: F401,F403
    from .qconfig import *  # noqa: F401,F403
