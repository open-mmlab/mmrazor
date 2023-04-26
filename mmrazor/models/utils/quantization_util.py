# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.utils import import_modules_from_strings


def pop_rewriter_function_record(rewriter_context, function_record_to_pop):
    """Delete user-specific rewriters from `RewriterContext._rewriter_manager`.

    We use the model which is rewritten by mmdeploy to build quantized models.
    However not all the functions rewritten by mmdeploy need to be rewritten in
    mmrazor. For example, mmdeploy rewrite
    `mmcls.models.classifiers.ImageClassifier.forward` and
    `mmcls.models.classifiers.BaseClassifier.forward` for deployment. But they
    can't be rewritten by mmrazor as ptq and qat are done in mmrazor. So to
    ensure ptq and qat proceed normally, we have to remove these record from
    `RewriterContext._rewriter_manager`.
    """
    function_record_backup = {}
    for record in function_record_to_pop:
        records = rewriter_context._rewriter_manager.function_rewriter. \
            _registry._rewrite_records
        if record in records:
            function_record_backup[record] = records.pop(record)
    return function_record_backup


def _check_valid_source(source):
    """Check if the source's format is valid."""
    if not isinstance(source, str):
        raise TypeError(f'source should be a str '
                        f'instance, but got {type(source)}')

    assert len(source.split('.')) > 1, \
        'source must have at least one `.`'


def str2class(str_inputs):
    clss = []
    if not isinstance(str_inputs, tuple) and not isinstance(str_inputs, list):
        str_inputs_list = [str_inputs]
    else:
        str_inputs_list = str_inputs
    for s_class in str_inputs_list:
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
    if isinstance(str_inputs, list):
        return clss
    elif isinstance(str_inputs, tuple):
        return tuple(clss)
    else:
        return clss[0]
