from mmrazor.registry import LOOPS
from mmengine.runner import EpochBasedTrainLoop

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
        if isinstance(dataloader, dict):
            # Determine whether or not different ranks use different seed.
            diff_rank_seed = runner._randomness_cfg.get(
                'diff_rank_seed', False)
            self.calibrate_dataloader = runner.build_dataloader(
                calibrate_dataloader, seed=runner.seed, diff_rank_seed=diff_rank_seed)
        else:
            self.calibrate_dataloader = calibrate_dataloader
        
        self.is_calibrate = True if calibrate_dataloader is not None else False
    
    def calibrate(self, calibrate_dataloader) -> None:
    
    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        
        self.runner.model.prepare()
        
        if self.is_calibrate:
            self.runner.model.state(1, 0)
            self.calibrate(self.calibrate_dataloader)
        
        self.runner.model.state(1, 1)

        while self._epoch < self._max_epochs:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()
        
        self.runner.model.convert()
        
        self.runner.val_loop.run()
        
        self.runner.call_hook('after_train')