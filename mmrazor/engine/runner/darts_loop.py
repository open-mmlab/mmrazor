# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Union

from mmengine.runner import EpochBasedTrainLoop, IterBasedTrainLoop
from torch.utils.data import DataLoader

from mmrazor.registry import LOOPS


@LOOPS.register_module()
class DartsEpochBasedTrainLoop(EpochBasedTrainLoop):
    """EpochBasedTrainLoop for `Darts <https://arxiv.org/abs/1806.09055>`_

    In Darts, Two dataloaders are needed in the training stage. One
    (`dataloader`) is used to train the supernet and update its weights,
    another(`mutator_dataloader`) is only used to train and update the
    parameters of the supernet's architecture setting. In
    `DartsEpochBasedTrainLoop`, these dataloaders will be combined as a
    special dataloader, whose `data_batch` will contain both of the
    dataloaders' `data_batch`.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or Dict):
            A dataloader object or a dict to build a dataloader for
            training the model.
        mutator_dataloader (Dataloader or Dict):
            A dataloader object or a dict to build a dataloader for
            training the parameters of model architecture.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[Dict, DataLoader],
                 mutator_dataloader: Union[Dict, DataLoader],
                 max_epochs: int,
                 val_begin: int = 1,
                 val_interval: int = 1) -> None:
        super().__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval)
        if isinstance(mutator_dataloader, dict):
            self.mutator_dataloader = runner.build_dataloader(
                mutator_dataloader, seed=runner.seed)
        else:
            self.mutator_dataloader = mutator_dataloader
        self.multi_loaders = [self.dataloader, self.mutator_dataloader]

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')
        self.runner.model.train()

        for idx, data_batch in enumerate(EpochMultiLoader(self.multi_loaders)):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1


@LOOPS.register_module()
class DartsIterBasedTrainLoop(IterBasedTrainLoop):
    """IterBasedTrainLoop for `Darts <https://arxiv.org/abs/1806.09055>`_

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or Dict):
            A dataloader object or a dict to build a dataloader.
        mutator_dataloader (Dataloader or Dict):
            A dataloader object or a dict to build a dataloader for
            training the parameters of model architecture.
        max_iter (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[Dict, DataLoader],
                 mutator_dataloader: Union[Dict, DataLoader],
                 max_iters: int,
                 val_begin: int = 1,
                 val_interval: int = 1000) -> None:
        super().__init__(runner, dataloader, max_iters, val_begin,
                         val_interval)
        if isinstance(mutator_dataloader, dict):
            self.mutator_dataloader = runner.build_dataloader(
                mutator_dataloader, seed=runner.seed)
        else:
            self.mutator_dataloader = mutator_dataloader
        multi_loaders = [self.dataloader, self.mutator_dataloader]
        self.multi_loaders = IterMultiLoader(multi_loaders)

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        while self._iter < self._max_iters:
            self.runner.model.train()

            data_batch = next(self.multi_loaders)  # type: ignore
            self.run_iter(data_batch)

            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')


class EpochMultiLoader:
    """Multi loaders based on epoch."""

    def __init__(self, dataloaders: List[DataLoader]):
        self._dataloaders = dataloaders
        self.iter_loaders = [iter(loader) for loader in self._dataloaders]

    @property
    def num_loaders(self):
        """The number of dataloaders."""
        return len(self._dataloaders)

    def __iter__(self):
        """Return self when executing __iter__."""
        return self

    def __next__(self):
        """Get the next iter's data of multiple loaders."""
        data = tuple([next(loader) for loader in self.iter_loaders])

        return data

    def __len__(self):
        """Get the length of loader."""
        return min([len(loader) for loader in self._dataloaders])


class IterMultiLoader:
    """Multi loaders based on iter."""

    def __init__(self, dataloaders: Union[List[DataLoader], DataLoader]):
        self._dataloaders = dataloaders if isinstance(dataloaders,
                                                      list) else [dataloaders]
        self.iter_loaders = [iter(loader) for loader in self._dataloaders]
        self._epoch = 0

    @property
    def epoch(self):
        """The property of the class."""
        return self._epoch

    @property
    def num_loaders(self):
        """The number of dataloaders."""
        return len(self._dataloaders)

    def __next__(self):
        """Get the next iter's data of multiple loaders."""
        try:
            data = tuple([next(loader) for loader in self.iter_loaders])
        except StopIteration:
            self._epoch += 1
            for loader in self._dataloaders:
                if hasattr(loader.sampler, 'set_epoch'):
                    loader.sampler.set_epoch(self._epoch)
            self.iter_loader = [iter(loader) for loader in self._dataloaders]
            data = tuple([next(loader) for loader in self.iter_loaders])

        return data

    def __len__(self):
        """Get the length of loader."""
        return min([len(loader) for loader in self._dataloaders])
