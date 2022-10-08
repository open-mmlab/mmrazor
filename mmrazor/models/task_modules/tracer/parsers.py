# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, Dict

from .path import (Path, PathConcatNode, PathConvNode, PathDepthWiseConvNode,
                   PathLinearNode, PathList, PathNormNode)


def _is_leaf_grad_fn(grad_fn):
    """Determine whether the current node is a leaf node."""
    if type(grad_fn).__name__ == 'AccumulateGrad':
        return True
    return False


def parse_conv(tracer, grad_fn, module2name, param2module, cur_path,
               result_paths, visited, shared_module):
    """Parse the backward of a conv layer.

    Example:
        >>> conv = nn.Conv2d(3, 3, 3)
        >>> pseudo_img = torch.rand(1, 3, 224, 224)
        >>> out = conv(pseudo_img)
        >>> out.grad_fn.next_functions
        ((None, 0), (<AccumulateGrad object at 0x0000020E405CBD88>, 0),
        (<AccumulateGrad object at 0x0000020E405CB588>, 0))
        >>> # op.next_functions[0][0] is None means this ThnnConv2DBackward
        >>> # op has no parents
        >>> # op.next_functions[1][0].variable is the weight of this Conv2d
        >>> # module
        >>> # op.next_functions[2][0].variable is the bias of this Conv2d
        >>> # module
    """
    leaf_grad_fn = grad_fn.next_functions[1][0]
    while not _is_leaf_grad_fn(leaf_grad_fn):
        leaf_grad_fn = leaf_grad_fn.next_functions[0][0]
    variable = leaf_grad_fn.variable
    param_id = id(variable)
    module = param2module[param_id]
    name = module2name[module]
    parent = grad_fn.next_functions[0][0]
    if module.in_channels == module.groups:
        cur_path.append(PathDepthWiseConvNode(name))
    else:
        cur_path.append(PathConvNode(name))
    # If a module is not a shared module and it has been visited during
    # forward, its parent modules must have been traced already.
    # However, a shared module will be visited more than once during
    # forward, so it is still need to be traced even if it has been
    # visited.
    if visited[name] and name not in shared_module:
        result_paths.append(copy.deepcopy(cur_path))
    else:
        visited[name] = True
        tracer.backward_trace(parent, module2name, param2module, cur_path,
                              result_paths, visited, shared_module)
    cur_path.pop(-1)


# todo: support parsing `MultiheadAttention` and user-defined matrix
#  multiplication
def parse_linear(tracer, grad_fn, module2name, param2module, cur_path,
                 result_paths, visited, shared_module):
    """Parse the backward of a conv layer.

    Example:
        >>> fc = nn.Linear(3, 3, bias=True)
        >>> input = torch.rand(3, 3)
        >>> out = fc(input)
        >>> out.grad_fn.next_functions
        ((<AccumulateGrad object at 0x0000020E405F75C8>, 0), (None, 0),
        (<TBackward object at 0x0000020E405F7D48>, 0))
        >>> # op.next_functions[0][0].variable is the bias of this Linear
        >>> # module
        >>> # op.next_functions[1][0] is None means this AddmmBackward op
        >>> # has no parents
        >>> # op.next_functions[2][0] is the TBackward op, and
        >>> # op.next_functions[2][0].next_functions[0][0].variable is
        >>> # the transpose of the weight of this Linear module
    """
    leaf_grad_fn = grad_fn.next_functions[-1][0].next_functions[0][0]
    while not _is_leaf_grad_fn(leaf_grad_fn):
        leaf_grad_fn = leaf_grad_fn.next_functions[0][0]
    variable = leaf_grad_fn.variable
    param_id = id(variable)
    module = param2module[param_id]
    name = module2name[module]
    parent = grad_fn.next_functions[-2][0]

    cur_path.append(PathLinearNode(name))
    # If a module is not a shared module and it has been visited during
    # forward, its parent modules must have been traced already.
    # However, a shared module will be visited more than once during
    # forward, so it is still need to be traced even if it has been
    # visited.
    if visited[name] and name not in shared_module:
        result_paths.append(copy.deepcopy(cur_path))
    else:
        visited[name] = True
        tracer.backward_trace(parent, module2name, param2module, cur_path,
                              result_paths, visited, shared_module)


def parse_cat(tracer, grad_fn, module2name, param2module, cur_path,
              result_paths, visited, shared_module):
    """Parse the backward of a concat operation.

    Example:
        >>> conv = nn.Conv2d(3, 3, 3)
        >>> pseudo_img = torch.rand(1, 3, 224, 224)
        >>> out1 = conv(pseudo_img)
        >>> out2 = conv(pseudo_img)
        >>> out = torch.cat([out1, out2], dim=1)
        >>> out.grad_fn.next_functions
        ((<ThnnConv2DBackward object at 0x0000020E405F24C8>, 0),
        (<ThnnConv2DBackward object at 0x0000020E405F2648>, 0))
        >>> # the length of ``out.grad_fn.next_functions`` is two means
        >>> # ``out`` is obtained by concatenating two tensors
    """
    parents = grad_fn.next_functions
    concat_id = '_'.join([str(id(p)) for p in parents])
    concat_id_list = [str(id(p)) for p in parents]
    concat_id_list.sort()
    concat_id = '_'.join(concat_id_list)
    name = f'concat_{concat_id}'

    visited[name] = True
    sub_path_lists = list()
    for _, parent in enumerate(parents):
        sub_path_list = PathList()
        tracer.backward_trace(parent, module2name, param2module, Path(),
                              sub_path_list, visited, shared_module)
        sub_path_lists.append(sub_path_list)
    cur_path.append(PathConcatNode(name, sub_path_lists))

    result_paths.append(copy.deepcopy(cur_path))
    cur_path.pop(-1)


def parse_norm(tracer, grad_fn, module2name, param2module, cur_path,
               result_paths, visited, shared_module):
    """Parse the backward of a concat operation.

    Example:
        >>> conv = nn.Conv2d(3, 3, 3)
        >>> pseudo_img = torch.rand(1, 3, 224, 224)
        >>> out1 = conv(pseudo_img)
        >>> out2 = conv(pseudo_img)
        >>> out = torch.cat([out1, out2], dim=1)
        >>> out.grad_fn.next_functions
        ((<ThnnConv2DBackward object at 0x0000020E405F24C8>, 0),
        (<ThnnConv2DBackward object at 0x0000020E405F2648>, 0))
        >>> # the length of ``out.grad_fn.next_functions`` is two means
        >>> # ``out`` is obtained by concatenating two tensors
    """
    leaf_grad_fn = grad_fn.next_functions[1][0]
    while not _is_leaf_grad_fn(leaf_grad_fn):
        leaf_grad_fn = leaf_grad_fn.next_functions[0][0]
    variable = leaf_grad_fn.variable
    param_id = id(variable)
    module = param2module[param_id]
    name = module2name[module]
    parent = grad_fn.next_functions[0][0]
    cur_path.append(PathNormNode(name))

    visited[name] = True
    tracer.backward_trace(parent, module2name, param2module, cur_path,
                          result_paths, visited, shared_module)
    cur_path.pop(-1)


DEFAULT_BACKWARD_TRACER: Dict[str, Callable] = {
    'ConvolutionBackward': parse_conv,
    'SlowConv2DBackward': parse_conv,
    'ThnnConv2DBackward': parse_conv,
    'CudnnConvolutionBackward': parse_conv,
    'MkldnnConvolutionBackward': parse_conv,
    'SlowConvDilated2DBackward': parse_conv,
    'ThAddmmBackward': parse_linear,
    'AddmmBackward': parse_linear,
    'MmBackward': parse_linear,
    'CatBackward': parse_cat,
    'ThnnBatchNormBackward': parse_norm,
    'CudnnBatchNormBackward': parse_norm,
    'NativeBatchNormBackward': parse_norm,
    'NativeGroupNormBackward': parse_norm
}
