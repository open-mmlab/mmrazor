# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.utils import import_modules_from_strings
from torch.nn.modules.batchnorm import _BatchNorm

import functools
from ..builder import DISTILLERS, MODELS, build_loss


class DistillRewriter():

    def __init__(self,
                 function,
                 dependent_module,
                 max_data_queue_length=1,
                 source='teacher',
                 target='student'):
        assert source in ['student', 'teacher']
        assert target in ['student', 'teacher']
        self.current_mode = 'eval'
        self.data_queue = list()
        self.max_data_queue_length = max_data_queue_length
        self.source = source
        self.target = target
        imported_module = import_modules_from_strings(  # noqa: F841
            dependent_module)
        raw_func = eval(f'imported_module.{function}')
        wrap_func = self.function_rewrite_wrapper(raw_func)  # noqa: F841
        exec(f'imported_module.{function} = wrap_func')

    def set_current_mode(self, mode):
        assert mode in ['student', 'teacher', 'eval']
        self.current_mode = mode

    def function_rewrite_wrapper(self, raw_func):

        def wrap_func(*args, **kwargs):
            if self.current_mode == self.target:
                assert len(self.data_queue) > 0
                outputs = self.data_queue.pop(0)
            elif self.current_mode == self.source:
                outputs = raw_func(*args, **kwargs)
                assert len(self.data_queue) < self.max_data_queue_length
                self.data_queue.append(outputs)
            elif self.current_mode == 'eval':
                outputs = raw_func(*args, **kwargs)
            else:
                raise RuntimeError
            return outputs

        return wrap_func


class Recorder():

    def __init__(self, 
                 model, 
                 inputs=None, 
                 outputs=None, 
                 weights=None, 
                 methods=None,
                 functions=None,
                 **kwargs) -> None:
        self.outputs = dict()
        self.inputs = dict()
        self.weights = dict()
        self.methods = dict()
        self.functions = dict()
        self.functions_cfg = functions
        self.origin_functions = dict()
        self.rewrite_functions = dict()
        self.recording = False

        if weights is not None:
            for param_name, param in model.named_parameters():
                if param_name in weights.sources:
                    self.weights[param_name] = param

        self.module2name = {}
        for module_name, module in model.named_modules():
            self.module2name[module] = module_name
        self.name2module = dict(model.named_modules())

        if outputs is not None:
            for module_name in outputs.sources:
                self.outputs[module_name] = list()
                module = self.name2module[module_name]
                module.register_forward_hook(self.forward_output_hook)

        if inputs is not None:
            for module_name in inputs.sources:
                self.inputs[module_name] = list()
                module = self.name2module[module_name]
                module.register_forward_hook(self.forward_input_hook)

        if methods is not None:
            for method_name, mapping_module in zip(methods.sources,
                                                methods.mapping_modules):
                imported_module = import_modules_from_strings(  # noqa: F841
                    mapping_module)
                raw_method = eval(f'imported_module.{method_name}')
                full_method_name = f'{mapping_module}.{method_name}'
                # if full_func_name == 'mmdet.core.anchor_inside_flags':
                self.methods[full_method_name] = list()
                wrap_method = self.method_hook_wrapper(  # noqa: F841
                    raw_method, full_method_name)
                exec(f'imported_module.{method_name} = wrap_method')

        if functions is not None:
            for func_name, mapping_module in zip(functions.sources,
                                                functions.mapping_modules):
                imported_module = import_modules_from_strings(  # noqa: F841
                    mapping_module)
                raw_func = eval(f'imported_module.{func_name}')
                 
                full_func_name = f'{mapping_module}.{func_name}' 
                self.functions[full_func_name] = list()
                wrap_func = self.func_hook_wrapper(  # noqa: F841
                    self, raw_func, full_func_name)
                                
                exec(f'imported_module.{func_name} = wrap_func')

    def reset_record_items(self):
        for key in self.outputs.keys():
            self.outputs[key] = list()
        for key in self.inputs.keys():
            self.inputs[key] = list()
        for key in self.functions.keys():
            self.functions[key] = list()

    def set_recording(self, recording):
        self.recording = recording

    def get_record_item(self, source_type, source, index=None):
        source_items = getattr(self, source_type)
        item = source_items[source]
        if index:
            assert isinstance(index, int)
            assert index < len(item)
            return item[index]
        else:
            return item

    def method_hook_wrapper(self, raw_method, method_name):
        @functools.wraps(raw_method)
        def wrap_method(*args, **kwargs):
            outputs = raw_method(*args, **kwargs)
            if self.recording:
                self.methods[method_name].append(outputs)
            return outputs

        return wrap_method

    @staticmethod
    def func_hook_wrapper(ctx, raw_func, func_name):
        @functools.wraps(raw_func)
        def wrap_func(*args, **kwargs):
            outputs = raw_func(*args, **kwargs)
            if ctx.recording:
                ctx.functions[func_name].append(outputs)
            return outputs

        return wrap_func

    def forward_input_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.recording:
            module_name = self.module2name[module]
            self.inputs[module_name].append(inputs)

    def forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.recording:
            module_name = self.module2name[module]
            self.outputs[module_name].append(outputs)

    def __enter__(self):
        self.recording = True
        self.reset_record_items()

    def __exit__(self, exc_type, exc_value, traceback):
        """Restore the function."""
        self.recording = False



@DISTILLERS.register_module()
class SingleTeacherDistillerV2(BaseModule):
    """Distiller with single teacher.

    Args:
        teacher (dict): The config dict for teacher.
        teacher_trainable (bool): Whether the teacher is trainable.
            Default: False.
        teacher_norm_eval (bool): Whether to set teacher's norm layers to eval
            mode, namely, freeze running stats (mean and var). Note: Effect on
            Batch Norm and its variants only. Default: True.
        components (dict): The details of the distillation. It usually includes
            the module names of the teacher and the student, and the losses
            used in the distillation.
    """

    def __init__(self,
                 teacher,
                 student_recorder_cfg,
                 teacher_recorder_cfg,
                 rewriters=tuple(),
                 teacher_trainable=False,
                 teacher_norm_eval=True,
                 components=tuple(),
                 **kwargs):
        super().__init__(**kwargs)
        self.teacher_trainable = teacher_trainable
        self.teacher_norm_eval = teacher_norm_eval
        self.teacher = self.build_teacher(teacher)

        self.student_recorder_cfg = student_recorder_cfg
        self.teacher_recorder_cfg = teacher_recorder_cfg

        self.components = components
        self.losses = nn.ModuleDict()

        for i, component in enumerate(self.components):
            loss_name = f'loss_{i}'
            self.losses[loss_name] = build_loss(component.loss)

        self.rewriters = list()
        for rewriter_cfg in rewriters:
            self.rewriters.append(DistillRewriter(**rewriter_cfg))

    def build_teacher(self, cfg):
        """Build a model from the `cfg`."""

        teacher = MODELS.build(cfg)

        return teacher

    def prepare_from_student(self, student):
        """Registers a global forward hook for each teacher module and student
        module to be used in the distillation.

        Args:
            student (:obj:`torch.nn.Module`): The student model to be used
                in the distillation.
        """

        
        if self.student_recorder_cfg is None:
            self.student_recorder = None
        else:
            self.student_recorder = Recorder(student.model,
                                         **self.student_recorder_cfg)

        if self.teacher_recorder_cfg is None:
            self.teacher_recorder = None
        else:
            self.teacher_recorder = Recorder(self.teacher,
                                         **self.teacher_recorder_cfg)

    def exec_teacher_forward(self, data):
        """Execute the teacher's forward function.

        After this function, the teacher's featuremaps will be saved in
        ``teacher_outputs``.
        """
        for rewriter in self.rewriters:
            rewriter.set_current_mode('teacher')

        if self.teacher_recorder is None:
            
            if self.teacher_trainable:
                output = self.teacher(**data)
            else:
                with torch.no_grad():
                    output = self.teacher(**data)
        else:
            with self.teacher_recorder:
                if self.teacher_trainable:
                    output = self.teacher(**data)
                else:
                    with torch.no_grad():
                        output = self.teacher(**data)
        for rewriter in self.rewriters:
            rewriter.set_current_mode('eval')
        return output

    def exec_student_forward(self, student, data):
        """Execute the teacher's forward function.

        After this function, the student's featuremaps will be saved in
        ``student_outputs``.
        """
        if self.student_recorder is None:
            for rewriter in self.rewriters:
                rewriter.set_current_mode('student')
            output = student(**data)

            for rewriter in self.rewriters:
                rewriter.set_current_mode('eval')
        else:
            with self.student_recorder:
                for rewriter in self.rewriters:
                    rewriter.set_current_mode('student')
                output = student(**data)

                for rewriter in self.rewriters:
                    rewriter.set_current_mode('eval')
        return output

    def train(self, mode=True):
        """Set distiller's forward mode."""
        super(SingleTeacherDistillerV2, self).train(mode)
        if mode and self.teacher_norm_eval:
            for m in self.teacher.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

    def compute_distill_loss(self, data=None):
        """Compute the distillation loss."""

        losses = dict()

        for i, component in enumerate(self.components):
            # Get the student's outputs.
            student_items = list()
            for item_cfg in component.student_items:
                item = self.student_recorder.get_record_item(**item_cfg)
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    student_items.append(item[0])
                else:
                    student_items.append(item)

            # Get the teacher's outputs.
            teacher_items = list()
            for item_cfg in component.teacher_items:
                item = self.teacher_recorder.get_record_item(**item_cfg)
                if isinstance(item, (list, tuple)) and len(item) == 1:
                    teacher_items.append(item[0])
                else:
                    teacher_items.append(item)

            loss_name = f'loss_{i}'
            loss_module = self.losses[loss_name]
            # TODO ugly implementation.
            # Pass the gt_label to loss function.
            # Only used by WSLD.
            loss_module.current_data = data
            losses[loss_name] = loss_module(*student_items, *teacher_items)
            loss_module.current_data = None

        return losses
