from mmrazor.registry import LOOPS
from mmrazor.models.task_modules import ModuleInputsRecorder, ModuleOutputsRecorder
from mmengine.runner import EpochBasedTrainLoop
from typing import Union, Optional, List, Tuple, Dict
from torch.utils.data import DataLoader
from utils import extract_subgraph, extract_blocks
import numpy as np

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
        super().__init__(runner, 
                         dataloader, 
                         max_epochs, 
                         val_begin, 
                         val_interval, 
                         dynamic_intervals)
        if isinstance(calibrate_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.calibrate_dataloader = runner.build_dataloader(
                calibrate_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.calibrate_dataloader = calibrate_dataloader
        
        self.is_calibrate = True if calibrate_dataloader is not None else False
        
        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model
    
    # TODO: to finish
    def calibrate(self, calibrate_dataloader) -> None:
        pass
    
    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        
        self.model.prepare()
        
        if self.is_calibrate:
            self.model.state(1, 0)
            self.calibrate(self.calibrate_dataloader)
        
        self.model.state(1, 1)

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
                 calibrate_dataloader=None,
                 reconstruction_cfg: Optional[Dict] = None,
                 fp16: bool = False):
        super().__init__(runner, 
                         dataloader, 
                         max_epochs, 
                         evaluator, 
                         fp16)
        if isinstance(calibrate_dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.calibrate_dataloader = runner.build_dataloader(
                calibrate_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.calibrate_dataloader = calibrate_dataloader
        
        self.is_calibrate = True if calibrate_dataloader is not None else False
        
        if self.runner.distributed:
            self.model = runner.model.module
        else:
            self.model = runner.model
    
    # TODO: to finish
    def calibrate(self, calibrate_dataloader) -> None:
        pass

    def save_inter_result(model, dataloader, slices, store_input=True, store_output=True):
        recorders = {}
        for s in slices:
            node_l, node_r = s[:2]
            if store_input:
                recorders[node_l.target + '_input'] = ModuleInputsRecorder(node_l.target)
            if store_output:
                recorders[node_r.target + '_output'] = MethodOutputsRecorder(node_r.target)
        manager = RecorderManager(recorders)
        manager.initialize(model)
        
        with torch.no_grad():
            with manager:
                for data in dataloader:
                    model(data)
        return manager

    def sub_reconstruction(graphmodule, manager, config):
        pass

    def reconstruction(self, graphmodule, calibrate_dataloader, config):
        assert isinstance(graphmodule, torch.fx.GraphModule)
        graphmodule_fp = graphmodule
        graphmodule_quant = copy.deepcopy(graphmodule)

        # get layers/blocks need to reconstructe
        slices = []
        if config['pattern'] == 'layer':
            slices = extract_layers(graphmodule, layer_types=_ADAROUND_SUPPORT_TYPE)
        elif config['pattern'] == 'block':
            slices = extract_blocks(graphmodule)
        else:
            # TODO: add remind
            raise NotImplementedError
        
        # save fp inputs and outputs of each layers
        manager_fp = self.save_inter_result(graphmodule_fp, self.calibrate_dataloader, slices)

        # extract subgraph_module
        for s in slices:
            sub_graphmodule = extract_subgraph(graphmodule_quant, s)
            manager_quant = self.save_inter_result(graphmodule_quant, self.calibrate_dataloader, [s], store_output=False)
            recorder_index = s[0].target + '_input'
            cached_inputs = manager_fp[recorder_index] if np.random.random() < config['prob'] else manager_quant[recorder_index]
            sub_reconstruction(sub_graphmodule, cached_inputs, cached_outputs, config)

        return graphmodule_quant
    
    def run(self) -> None:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')

        self.model.prepare()
        
        if self.is_calibrate:
            self.model.state(1, 0)
            self.calibrate(self.calibrate_dataloader)
        
        self.model.state(1, 1)

        self.reconstruction()

        self.model.convert()

        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics
        
