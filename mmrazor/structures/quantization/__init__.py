# Copyright (c) OpenMMLab. All rights reserved.
from .backend_default_qconfigs import CheckArgs, DefalutQconfigs, SupportQtypes
from .qscheme import QuantizeScheme

__all__ = ['QuantizeScheme', 'DefalutQconfigs', 'SupportQtypes', 'CheckArgs']
