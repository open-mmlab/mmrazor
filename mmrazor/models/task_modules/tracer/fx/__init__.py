# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmrazor import digit_version

if digit_version(torch.__version__) >= digit_version('1.13.0'):
    from .custom_tracer import CustomTracer  # noqa: F401,F403
    from .custom_tracer import UntracedMethodRegistry  # noqa: F401,F403
    from .custom_tracer import build_graphmodule  # noqa: F401,F403
    from .custom_tracer import custom_symbolic_trace  # noqa: F401,F403
    from .graph_utils import del_fakequant_after_function  # noqa: F401,F403
    from .graph_utils import del_fakequant_after_method  # noqa: F401,F403
    from .graph_utils import del_fakequant_after_module  # noqa: F401,F403
    from .graph_utils import del_fakequant_after_op  # noqa: F401,F403
    from .graph_utils import del_fakequant_before_function  # noqa: F401,F403
    from .graph_utils import del_fakequant_before_method  # noqa: F401,F403
    from .graph_utils import del_fakequant_before_module  # noqa: F401,F403
    from .graph_utils import del_fakequant_before_op  # noqa: F401,F403

    __all__ = [
        'CustomTracer', 'UntracedMethodRegistry', 'custom_symbolic_trace',
        'build_graphmodule', 'del_fakequant_before_module',
        'del_fakequant_after_module', 'del_fakequant_after_function',
        'del_fakequant_before_function', 'del_fakequant_after_op',
        'del_fakequant_before_op', 'del_fakequant_before_method',
        'del_fakequant_after_method'
    ]
