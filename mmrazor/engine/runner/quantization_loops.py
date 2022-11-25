# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from mmengine.evaluator import Evaluator
from mmengine.registry import MODELS
from mmengine.runner import EpochBasedTrainLoop, TestLoop, ValLoop, autocast
from torch.ao.quantization import disable_observer
from torch.nn.intrinsic.qat import freeze_bn_stats
from torch.utils.data import DataLoader

from mmrazor.models.task_modules import (ModuleInputsRecorder,
                                         ModuleOutputsRecorder,
                                         RecorderManager)
from mmrazor.registry import LOOPS
from .utils import extract_blocks, extract_layers, extract_subgraph

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)


@LOOPS.register_module()
class QATEpochBasedLoop(EpochBasedTrainLoop):
    """`EpochBasedLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        disable_observer_begin (int): The number of total epochs to update
            observers.
        freeze_bn_begin (int): The number of total epochs to update batch norm
            stats.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            disable_observer_begin: int = 3,
            freeze_bn_begin: int = 3,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval, dynamic_intervals)

        self.disable_observer_begin = disable_observer_begin
        self.freeze_bn_begin = freeze_bn_begin

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        qat_metrics = dict()
        for key, value in metrics.items():
            qat_key = 'qat.' + key
            ori_key = 'original.' + key
            qat_metrics[qat_key] = value
            self.runner.message_hub.log_scalars.pop(f'val/{ori_key}', None)

        while self._epoch < self._max_epochs:
            # state: observer_enabled, fakequant_enabled
            self.runner.model.state = (True, True)
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                # observer disabled during evaluation
                self.runner.model.state = (False, True)
                self.runner.model.sync_param()
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        # TODO freeze bn
        if self._epoch >= self.disable_observer_begin:
            self.runner.model.apply(disable_observer)

        if self._epoch >= self.freeze_bn_begin:
            self.runner.model.apply(freeze_bn_stats)

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class QATValLoop(ValLoop):
    """`ValLoop` for `QuantizationAwareTraining`

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader, evaluator, fp16)
        if self.runner.distributed:
            assert hasattr(self.runner.model.module, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.module.data_preprocessor
            self.architecture = self.runner.model.module.architecture
            self.architecture.data_preprocessor = data_preprocessor

        else:
            assert hasattr(self.runner.model, 'architecture')
            # TODO: remove hard code after mmcls add data_preprocessor
            data_preprocessor = self.runner.model.data_preprocessor
            self.architecture = self.runner.model.architecture
            self.architecture.data_preprocessor = data_preprocessor

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, self.runner.model)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        qat_metrics = dict()
        for key, value in metrics.items():
            qat_key = 'qat.' + key
            ori_key = 'original.' + key
            qat_metrics[qat_key] = value
            self.runner.message_hub.log_scalars.pop(f'val/{ori_key}', None)

        self.runner.call_hook('after_val_epoch', metrics=qat_metrics)

        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch, self.architecture)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        qat_metrics = dict()
        for key, value in metrics.items():
            qat_key = 'qat.' + key
            ori_key = 'original.' + key
            qat_metrics[ori_key] = value
            self.runner.message_hub.log_scalars.pop(f'val/{qat_key}', None)

        self.runner.call_hook('after_val_epoch', metrics=qat_metrics)

        self.runner.call_hook('after_val')
        return qat_metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict], model):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class PTQLoop(TestLoop):
    """`TestLoop` for Post Training Quantization.

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool, optional): Enable FP16 training mode. Defaults to False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        self.runner.model.state = (True, False)

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

        # todo: hard code to save checkpoint on disk
        self.runner.save_checkpoint(
            self.runner.work_dir,
            'checkpoint_after_ptq.pth',
            file_client_args=None,
            save_optimizer=False,
            save_param_scheduler=False)

        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement

        outputs = self.runner.model.calibrate_step(data_batch)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


# TODO refactor to supoort DDP
@LOOPS.register_module()
class AdaRoundLoop(TestLoop):
    """`TestLoop` for Post Training Quantization.

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        calibrate_dataloader (Dataloader or dict, optional): A dataloader
            object or a dict to build a dataloader for calibration. Defaults
            to None.
        batch_num (Optional[int], optional): Total calibration batches.
            Defaults to None.
        reconstruction_cfg (Optional[Dict], optional): Model reconstruction
            configuration. Defaults to None.
        fp16 (bool, optional): Enable FP16 training mode. Defaults to False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)

    def run(self) -> None:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        self.runner.model.state = (1, 0)

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')

        # todo: hard code to save checkpoint on disk
        self.runner.save_checkpoint(
            self.runner.work_dir,
            'checkpoint_after_ptq.pth',
            file_client_args=None,
            save_optimizer=False,
            save_param_scheduler=False)

        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement

        outputs = self.runner.model.calibrate_step(data_batch)

        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


# TODO refactor to supoort DDP
@LOOPS.register_module()
class AdaRoundLoop(TestLoop):
    """`TestLoop` for Post Training Quantization.

    Args:
        runner (Runner): A reference of runner
        dataloader (Dataloader or dict): An iterator to generate one batch of
            dataset each iteration.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        calibrate_dataloader (Dataloader or dict, optional): A dataloader
            object or a dict to build a dataloader for calibration. Defaults
            to None.
        batch_num (Optional[int], optional): Total calibration batches.
            Defaults to None.
        reconstruction_cfg (Optional[Dict], optional): Model reconstruction
            configuration. Defaults to None.
        fp16 (bool, optional): Enable FP16 training mode. Defaults to False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 calibrate_dataloader: Optional[Union[DataLoader,
                                                      Dict]] = None,
                 batch_num: Optional[int] = None,
                 reconstruction_cfg: Optional[Dict] = None,
                 fp16: bool = False):
        super().__init__(runner, dataloader, evaluator, fp16)
        if isinstance(calibrate_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.calibrate_dataloader = runner.build_dataloader(
                calibrate_dataloader,
                seed=runner.seed,
                diff_rank_seed=diff_rank_seed)
        else:
            self.calibrate_dataloader = calibrate_dataloader

        self.is_calibrate = True if calibrate_dataloader is not None else False

        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model

        self.batch_num = batch_num
        self.config = reconstruction_cfg

    def calibrate(self, calibrate_dataloader) -> None:
        self.model.eval()
        with torch.no_grad():
            for i, batch_data in enumerate(calibrate_dataloader):
                if self.batch_num and i >= self.batch_num:
                    break
                self.model.calib_step(batch_data)

    def _save_inter_result(self,
                           model,
                           dataloader,
                           slices,
                           store_input=True,
                           store_output=True):
        recorders = {}
        for s in slices:
            node_l, node_r = s[:2]
            if store_input:
                recorders[node_l.target + '_input'] = ModuleInputsRecorder(
                    node_l.target)
            if store_output:
                recorders[node_r.target + '_output'] = ModuleOutputsRecorder(
                    node_r.target)
        manager = RecorderManager(recorders)
        manager.initialize(model)

        with torch.no_grad():
            with manager:
                for i, batch_data in enumerate(dataloader):
                    if self.batch_num and i >= self.batch_num:
                        break
                    batch_data = self.model.data_preprocessor(
                        batch_data, False)
                    model(**batch_data)
        return manager

    def sub_reconstruction(self, graphmodule, input_recorder, output_recorder,
                           config):
        w_para = []
        for layer in graphmodule.modules():
            # import pdb
            # pdb.set_trace()
            if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                weight_fake_quant = layer.weight_fake_quant
                weight_fake_quant.init(layer.weight.data)
                w_para += [weight_fake_quant.alpha]

        w_opt = torch.optim.Adam(w_para)
        loss_func = MODELS.build(config.loss)

        for _ in range(config.loss.iters):
            w_opt.zero_grad()

            data_size = len(input_recorder.data_buffer)
            data_index = np.random.randint(0, data_size)
            out_quant = graphmodule(
                input_recorder.get_recorder_data(data_index))
            out_fp = output_recorder.get_recorder_data(data_index)
            err = loss_func(graphmodule, out_quant, out_fp)
            err.backward()
            w_opt.step()

        for layer in graphmodule.modules():
            if isinstance(layer, _ADAROUND_SUPPORT_TYPE):
                weight_fake_quant = layer.weight_fake_quant
                layer.weight.data = weight_fake_quant.get_hard_value(
                    layer.weight.data)
                weight_fake_quant.adaround = False
            if isinstance(layer, torch.quantization.FakeQuantize) and hasattr(
                    layer, 'prob'):
                # recover to promise that drop activation quantization only
                # occurs at reconstruction phase
                layer.prob = 1.0

    def reconstruction(self, graphmodule, calibrate_dataloader, config):
        assert isinstance(graphmodule, torch.fx.GraphModule)
        graphmodule_fp = graphmodule
        graphmodule_quant = copy.deepcopy(graphmodule)

        # get layers/blocks need to reconstructe
        slices = []
        if config.pattern == 'layer':
            slices = extract_layers(
                graphmodule, layer_types=_ADAROUND_SUPPORT_TYPE)
        elif config.pattern == 'block':
            slices = extract_blocks(graphmodule)
        else:
            # TODO: add remind
            raise NotImplementedError

        # save fp inputs and outputs of each layers
        manager_fp = self._save_inter_result(graphmodule_fp,
                                             self.calibrate_dataloader, slices)

        # extract subgraph_module
        for s in slices:
            sub_graphmodule = extract_subgraph(graphmodule_quant, s)
            manager_quant = self._save_inter_result(
                graphmodule_quant,
                self.calibrate_dataloader, [s],
                store_output=False)
            input_index = s[0].target + '_input'
            output_index = s[1].target + '_output'
            input_recorder = manager_quant.get_recorder(input_index)
            output_recorder = manager_fp.get_recorder(output_index)
            self.sub_reconstruction(sub_graphmodule, input_recorder,
                                    output_recorder, config)

        return graphmodule_quant

    def run(self) -> None:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        self.model.eval()
        self.model.prepare()

        if self.is_calibrate:
            self.model.state = (1, 0)
            self.calibrate(self.calibrate_dataloader)

        self.model.state = (1, 1)

        if self.config is not None:
            self.model.architecture = self.reconstruction(
                self.model.architecture, self.calibrate_dataloader,
                self.config)

        self.model.convert()

        self.model.eval()
        from torch.onnx import OperatorExportTypes
        dummy_input = torch.randn([1, 3, 224, 224])
        onnx_path = os.path.join(self.runner.work_dir, 'quantizied.onnx')
        torch.onnx.export(
            self.model.architecture,
            dummy_input,
            onnx_path,
            opset_version=11,
            operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK)

        self.runner.call_hook('after_test')
