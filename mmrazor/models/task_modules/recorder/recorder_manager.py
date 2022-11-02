# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional

from torch import nn

from mmrazor.registry import TASK_UTILS
from .base_recorder import BaseRecorder


class RecorderManager:
    """Various types recorders' manager. The ``RecorderManager`` also is a
    context manager, managing various types of Recorder. When entering the
    ``RecorderManager``, all recorders managed by it will be started.

    Note:
        The recorders will be initialized in the ``RecorderManager`` by
        default. If you want to just use a recorder without the
        ``RecorderManager``, you need to initialize it first.

    Args:
        recorders (dict, optional): All recorders' config.


    Examples:
        >>> # Below code in toy_module.py
        >>> import random
        >>> class Toy():
        ...     def toy_func(self):
        ...         return random.randint(0, 1000)

        >>> # Below code in main.py
        >>> from torch import nn
        >>> from toy_module import Toy

        >>> class ToyModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.conv1 = nn.Conv2d(1,1,1)
        ...         self.conv2 = nn.Conv2d(1,1,1)
        ...         self.toy = Toy()
        ...     def forward(self, x):
        ...         return self.conv2(self.conv1(x)) + self.toy.toy_func()

        >>> model = ToyModel()
        >>> [ name for name,_ in model.named_modules() ]
        ['conv1', 'conv2']

        >>> conv1_rec = ModuleOutputsRecorder('conv1')
        >>> conv2_rec = ModuleOutputsRecorder('conv2')
        >>> func_rec = MethodOutputsRecorder('toy_module.Toy.toy_func')
        >>> manager = RecorderManager(
        ...             {'conv1_rec': conv1_rec ,
        ...              'conv2_rec': conv2_rec,
        ...              'func_rec': func_rec})
        >>> manager.initialize(model)

        >>> with manager:
        ...     res = model(torch.ones(1,1,1,1))
        >>> res
        tensor([[[[22.9534]]]])

        >>> conv2_data = manager.get_recorder('conv2_rec').get_record_data()
        >>> conv2_data
        tensor([[[[0.9534]]]])

        >>> func_data = manager.get_recorder('func_rec').get_record_data()
        >>> func_data
        22

        >>> res.sum() == (conv2_data + func_data).sum()
        True
    """

    def __init__(self, recorders: Optional[Dict] = None) -> None:

        self._recorders: Dict[str, BaseRecorder] = dict()
        if recorders:
            for name, cfg in recorders.items():
                recorder_cfg = copy.deepcopy(cfg)
                recorder_type = cfg['type']
                recorder_type_ = recorder_type + 'Recorder'

                recorder_cfg['type'] = recorder_type_
                recorder = TASK_UTILS.build(recorder_cfg)

                self._recorders[name] = recorder

    @property
    def recorders(self) -> Dict[str, BaseRecorder]:
        """dict: all recorders."""
        return self._recorders

    def get_recorder(self, recorder: str) -> BaseRecorder:
        """Get the corresponding recorder according to the name."""
        return self.recorders[recorder]

    def initialize(self, model: nn.Module):
        """Init all recorders.

        Args:
            model (nn.Module): The model which need to record intermediate
                results.
        """
        for recorder in self.recorders.values():
            recorder.initialize(model)

    def __enter__(self):
        """Enter the context manager."""
        for recorder in self.recorders.values():
            recorder.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        for recorder in self.recorders.values():
            recorder.__exit__(exc_type, exc_value, traceback)
