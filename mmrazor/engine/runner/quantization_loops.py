# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mmengine.evaluator import Evaluator
from mmengine.registry import MODELS
from mmengine.runner import EpochBasedTrainLoop, TestLoop
from torch.utils.data import DataLoader

from mmrazor.models.task_modules import (ModuleInputsRecorder,
                                         ModuleOutputsRecorder,
                                         RecorderManager)
from mmrazor.registry import LOOPS
from .utils import extract_blocks, extract_layers, extract_subgraph

_ADAROUND_SUPPORT_TYPE = (torch.nn.Conv2d, torch.nn.Linear)


@LOOPS.register_module()
class QATEpochBasedLoop(EpochBasedTrainLoop):

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            calibrate_dataloader=None,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval, dynamic_intervals)
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

    def calibrate(self, calibrate_dataloader) -> None:
        self.model.eval()
        with torch.no_grad():
            for batch_data in calibrate_dataloader:
                self.model(batch_data)

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')

        self.model.eval()
        self.model.prepare()

        if self.is_calibrate:
            self.model.state = (1, 0)
            self.calibrate(self.calibrate_dataloader)

        self.model.state = (1, 1)

        while self._epoch < self._max_epochs:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.model.convert()

        self.runner.val_loop.run()

        self.runner.call_hook('after_train')


@LOOPS.register_module()
class PTQLoop(TestLoop):

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

        for i in range(config.loss.iters):
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
        # dummy_input = torch.randn([1, 3, 224, 224]).cuda()
        # onnx_path = os.path.join(self.runner.work_dir, 'fp.onnx')
        # torch.onnx.export(
        #     self.model.architecture,
        #     dummy_input,
        #     onnx_path,
        #     opset_version=11,
        #     # operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK
        # )
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

        # self.model.eval()
        # for idx, data_batch in enumerate(self.dataloader):
        #     self.run_iter(idx, data_batch)

        # # compute metrics
        # metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

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

        # self.runner.save_checkpoint(
        #     out_dir=self.runner.work_dir,
        #     filename='quantizied.pth',
        #     save_optimizer=False,
        #     save_param_scheduler=False
        # )

        # self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        # return metrics
