# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from typing import Any, Dict, List, Optional, Set

import torch
import torch.distributed as dist
from mmengine.utils import import_modules_from_strings

def _check_valid_source(source):
        """Check if the source's format is valid."""
        if not isinstance(source, str):
            raise TypeError(f'source should be a str '
                            f'instance, but got {type(source)}')

        assert len(source.split('.')) > 1, \
            'source must have at least one `.`'

def str2class(str_list):
    clss = []
    if not isinstance(str_list, list):
        return clss
    for s_class in str_list:
        _check_valid_source(s_class)
        mod_str = '.'.join(s_class.split('.')[:-1])
        cls_str = s_class.split('.')[-1]
        try:
            mod = import_modules_from_strings(mod_str)
        except ImportError:
            raise ImportError(f'{mod_str} is not imported correctly.')
        imported_cls: type = getattr(mod, cls_str)
        if not isinstance(imported_cls, type):
            raise TypeError(f'{cls_str} should be a type '
                            f'instance, but got {type(imported_cls)}')
        clss.append(imported_cls)
    return clss





