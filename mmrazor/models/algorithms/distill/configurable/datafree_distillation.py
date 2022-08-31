# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmengine.optim import OPTIMIZERS, OptimWrapper
from mmengine.runner import load_checkpoint

from mmrazor.models.utils import add_prefix, set_requires_grad
from mmrazor.registry import MODELS
from ...base import BaseAlgorithm


@MODELS.register_module()
class DataFreeDistillation(BaseAlgorithm):
    """Algorithm for data-free teacher-student distillation Typically, the
    teacher is a pretrained model and the student is a small model trained on
    the generator's output. The student is trained to mimic the behavior of the
    teacher. The generator is trained to generate images that are similar to
    the real images.

    Args:
        distiller (dict): The config dict for built distiller.
        generator_distiller (dict): The distiller collecting outputs & losses
            to update the generator.
        teachers (dict[str, dict]): The dict of config dict for teacher models
            and their ckpt_path (optional).
        generator (dictl): The config dict for built distiller generator.
        student_iter (int): The number of student steps in train_step().
            Defaults to 1.
        student_train_first (bool): Whether to train student in first place.
            Defaults to False.
    """

    def __init__(self,
                 distiller: dict,
                 generator_distiller: dict,
                 teachers: Dict[str, Dict[str, dict]],
                 generator: dict,
                 student_iter: int = 1,
                 student_train_first: bool = False,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.student_iter = student_iter
        self.student_train_first = student_train_first
        self.distiller = MODELS.build(distiller)
        self.generator_distiller = MODELS.build(generator_distiller)

        if not isinstance(teachers, Dict):
            raise TypeError('teacher should be a `dict` but got '
                            f'{type(teachers)}')

        self.teachers = nn.ModuleDict()
        for teacher_name, cfg in teachers.items():
            self.teachers[teacher_name] = MODELS.build(cfg['build_cfg'])
            if 'ckpt_path' in cfg:
                # avoid loaded parameters be overwritten
                self.teachers[teacher_name].init_weights()
                _ = load_checkpoint(self.teachers[teacher_name],
                                    cfg['ckpt_path'])
            self.teachers[teacher_name].eval()
            set_requires_grad(self.teachers[teacher_name], False)

        if not isinstance(generator, Dict):
            raise TypeError('generator should be a `dict` instance, but got '
                            f'{type(generator)}')
        self.generator = MODELS.build(generator)

        # In ``DataFreeDistiller``, the recorder manager is just
        # constructed, but not really initialized yet.
        self.distiller.prepare_from_student(self.student)
        self.distiller.prepare_from_teacher(self.teachers)
        self.generator_distiller.prepare_from_student(self.student)
        self.generator_distiller.prepare_from_teacher(self.teachers)

    @property
    def student(self) -> nn.Module:
        """Alias for ``architecture``."""
        return self.architecture

    def train_step(self, data: Dict[str, List[dict]],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Train step for DataFreeDistillation.

        Args:
            data (Dict[str, List[dict]]): Data sampled by dataloader.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.
        """
        log_vars = dict()
        for _, teacher in self.teachers.items():
            teacher.eval()

        if self.student_train_first:
            _, dis_log_vars = self.train_student(data,
                                                 optim_wrapper['architecture'])
            _, generator_loss_vars = self.train_generator(
                data, optim_wrapper['generator'])
        else:
            _, generator_loss_vars = self.train_generator(
                data, optim_wrapper['generator'])
            _, dis_log_vars = self.train_student(data,
                                                 optim_wrapper['architecture'])

        log_vars.update(dis_log_vars)
        log_vars.update(generator_loss_vars)
        return log_vars

    def train_student(
            self, data: Dict[str, List[dict]], optimizer: OPTIMIZERS
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train step for the student model.

        Args:
            data (Dict[str, List[dict]]): Data sampled by dataloader.
            optimizer (OPTIMIZERS): The optimizer to update student.
        """
        log_vars = dict()
        batch_size = len(data['inputs'])

        for _ in range(self.student_iter):
            fakeimg_init = torch.randn(
                (batch_size, self.generator.module.latent_dim))
            fakeimg = self.generator(fakeimg_init, batch_size).detach()

            with optimizer.optim_context(self.student):
                pseudo_data = self.data_preprocessor(data, True)
                pseudo_data_samples = pseudo_data['data_samples']
                # recorde the needed information
                with self.distiller.student_recorders:
                    _ = self.student(fakeimg, pseudo_data_samples, mode='loss')
                with self.distiller.teacher_recorders, torch.no_grad():
                    for _, teacher in self.teachers.items():
                        _ = teacher(fakeimg, pseudo_data_samples, mode='loss')
                loss_distill = self.distiller.compute_distill_losses()

            distill_loss, distill_log_vars = self.parse_losses(loss_distill)
            optimizer.update_params(distill_loss)
        log_vars = dict(add_prefix(distill_log_vars, 'distill'))

        return distill_loss, log_vars

    def train_generator(
            self, data: Dict[str, List[dict]], optimizer: OPTIMIZERS
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train step for the generator.

        Args:
            data (Dict[str, List[dict]]): Data sampled by dataloader.
            optimizer (OPTIMIZERS): The optimizer to update generator.
        """
        batch_size = len(data['inputs'])
        fakeimg_init = torch.randn(
            (batch_size, self.generator.module.latent_dim))
        fakeimg = self.generator(fakeimg_init, batch_size)

        with optimizer.optim_context(self.generator):
            pseudo_data = self.data_preprocessor(data, True)
            pseudo_data_samples = pseudo_data['data_samples']
            # recorde the needed information
            with self.generator_distiller.student_recorders:
                _ = self.student(fakeimg, pseudo_data_samples, mode='loss')
            with self.generator_distiller.teacher_recorders:
                for _, teacher in self.teachers.items():
                    _ = teacher(fakeimg, pseudo_data_samples, mode='loss')
            loss_generator = self.generator_distiller.compute_distill_losses()

        generator_loss, generator_loss_vars = self.parse_losses(loss_generator)
        optimizer.update_params(generator_loss)
        log_vars = dict(add_prefix(generator_loss_vars, 'generator'))

        return generator_loss, log_vars


@MODELS.register_module()
class DAFLDataFreeDistillation(DataFreeDistillation):

    def train_step(self, data: Dict[str, List[dict]],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """DAFL train step.

        Args:
            data (Dict[str, List[dict]): Data sampled by dataloader.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.
        """
        log_vars = dict()
        batch_size = len(data['inputs'])

        for _, teacher in self.teachers.items():
            teacher.eval()

        # fakeimg initialization and revised by generator.
        fakeimg_init = torch.randn(
            (batch_size, self.generator.module.latent_dim))
        fakeimg = self.generator(fakeimg_init, batch_size)
        pseudo_data = self.data_preprocessor(data, True)
        pseudo_data_samples = pseudo_data['data_samples']

        with optim_wrapper['generator'].optim_context(self.generator):
            # recorde the needed information
            with self.generator_distiller.student_recorders:
                _ = self.student(fakeimg, pseudo_data_samples, mode='loss')
            with self.generator_distiller.teacher_recorders:
                for _, teacher in self.teachers.items():
                    _ = teacher(fakeimg, pseudo_data_samples, mode='loss')
            loss_generator = self.generator_distiller.compute_distill_losses()

        generator_loss, generator_loss_vars = self.parse_losses(loss_generator)
        log_vars.update(add_prefix(generator_loss_vars, 'generator'))

        with optim_wrapper['architecture'].optim_context(self.student):
            # recorde the needed information
            with self.distiller.student_recorders:
                _ = self.student(
                    fakeimg.detach(), pseudo_data_samples, mode='loss')
            with self.distiller.teacher_recorders, torch.no_grad():
                for _, teacher in self.teachers.items():
                    _ = teacher(
                        fakeimg.detach(), pseudo_data_samples, mode='loss')
            loss_distill = self.distiller.compute_distill_losses()

        distill_loss, distill_log_vars = self.parse_losses(loss_distill)
        log_vars.update(add_prefix(distill_log_vars, 'distill'))

        optim_wrapper['generator'].update_params(generator_loss)
        optim_wrapper['architecture'].update_params(distill_loss)

        return log_vars
