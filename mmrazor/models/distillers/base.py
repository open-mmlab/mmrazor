# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from mmcv.runner import BaseModule
from mmcv.utils import import_modules_from_strings


def function_wrapper(ctx, method, method_str):
    """Pass teacher's outputs to student."""

    def wrapper(*args, **kwargs):
        # record inputs
        ctx.method_args[method_str] = args
        ctx.method_kwargs[method_str] = kwargs
        # TODO cover more usecases, not only pass teacher's outputs to
        # student.
        if ctx.is_teacher:
            # execute the raw function
            outputs = method(*args, **kwargs)
            # record outputs
            ctx.method_return[method_str] = outputs
        else:
            # modify student's outputs to be same with teacher
            outputs = ctx.method_return[method_str]

        return outputs

    return wrapper


class FunctionContext():
    """Function context manager for rewrite function.

    Args:
        ctx (ConversionContext): The distiller's overall context manager.
        method (str): The name of the function to rewrite.
    """

    def __init__(self, ctx, method, import_module=None):
        self.ctx = ctx

        self.import_module = import_modules_from_strings(import_module)
        self.method_str = method
        self.method_exec_str = f'self.import_module.{method}'

    def _set_method(self, method):
        """Modify a function."""
        exec(f'{self.method_exec_str} = method')

    def __enter__(self):
        """Rewrite the function."""
        self.method_impl = eval(self.method_exec_str)

        if self.method_impl:
            self._set_method(
                function_wrapper(self.ctx, self.method_impl, self.method_str,
                                 self.align_mode))

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the function."""
        if self.method_impl:
            self._set_method(self.method_impl)


class ConversionContext():
    """Context manager for record functions' inputs or outputs."""

    def __init__(self, hooks):
        # save functions' inputs
        self.method_args = dict()
        self.method_kwargs = dict()
        # save functions' outputs
        self.method_return = dict()

        # Each function will have a sub context manager, the function will be
        # rewritten when enter the sub context manager.
        self.hooks = []
        self.is_teacher = True
        for hook in hooks:
            self.hooks.append(FunctionContext(self, **hook))

    def __enter__(self):
        """Enter every sub context managers."""
        for hook in self.hooks:
            hook.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit every sub context managers."""
        for hook in self.hooks:
            hook.__exit__(exc_type, exc_value, traceback)


class BaseDistiller(BaseModule, metaclass=ABCMeta):
    """Base Distiller.

    In the distillation algorithm, some intermediate results of the teacher
    need to be obtained and passed to the student.

    For nn.Module's outputs, obtained by pytorch forward hook.
    For python function's outputs, obtained by a specific context manager.

    Args:
        align_methods (dict): The details of the functions which outputs need
        to be obtained.
    """

    def __init__(self, align_methods=None, **kwargs):
        super(BaseDistiller, self).__init__(**kwargs)

        if align_methods is None:
            self.context_manager = None
        else:
            # To obtain the python function's outputs, there will build a
            # specific context manager. When enter the context manager, the
            # functions will be rewrite. The context manager could record
            # inputs or outputs of the functions , and pass from teachr to
            # student. When exit the context manager, the rewritten functions
            # will restore.
            self.context_manager = ConversionContext(align_methods)

    @abstractmethod
    def prepare_from_student(self, student):
        """Register forward hooks to students and teachers."""
        pass

    @abstractmethod
    def teacher_forward_output_hook(self, module, inputs, outputs):
        """Save the teacher output."""
        pass

    @abstractmethod
    def student_forward_output_hook(self, module, inputs, outputs):
        """Save the student output."""
        pass

    def reset_ctx_teacher_mode(self, mode=True):
        if self.context_manager is not None:
            self.context_manager.is_teacher = mode

    @abstractmethod
    def exec_teacher_forward(self, data):
        """Execute the teacher's forward function."""
        pass

    @abstractmethod
    def exec_student_forward(self, student, data):
        """Execute the student's forward function."""
        pass

    @abstractmethod
    def compute_distill_loss(self, data):
        """Compute distill loss according teacher's outputs and student's
        outputs."""
        pass
