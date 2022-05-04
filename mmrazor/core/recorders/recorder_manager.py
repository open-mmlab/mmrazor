# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Any, Dict, Optional

from ..builder import build_recorder
from .base_recorder import BaseRecorder


class RecorderManager():
    """Various types recorders' manager. The ``RecorderManager`` also is a
    context manager, managing various types of Recorder. When entering the
    ``RecorderManager``, all recorders managed by it will be started.

    Note:
        The recorders will be initialized in the ``RecorderManager`` by
        default. If you want to just use a recorder without the
        ``RecorderManager``, you need to initialize it first.

    Note:
        Recorders are managed by type. There can only be at most one recorder
        of each type. If you have multiple same type's source data, just need
        to add ``sources`` in your config.

    Args:
        recorders (list(dict)): All recorders' config.

    Examples:
            >>> from torch.nn import Module, ModuleList, Conv2d
            >>> from mmrazor.core import RecorderManager

            >>> # example module
            >>> class RepeatModule(Module):
            >>>     def __init__(self) -> None:
            >>>         super().__init__()
            >>>     def forward(self, x):
            >>>         outputs = list()
            >>>         for i in range(3):
            >>>             outputs.append(x * i)
            >>>         return outputs

            >>> # example model
            >>> class Model(Module):
            >>>     def __init__(self) -> None:
            >>>        super().__init__()
            >>>        self.repeat_module1 = RepeatModule()
            >>>        self.repeat_module2 = RepeatModule()
            >>>     def forward(self, x):
            >>>        out = self.repeat_module1(x)
            >>>        out = self.repeat_module2(x)
            >>>        return out

            >>> # example recorder configs
            >>> recorder_cfgs = [
            >>>     dict(
            >>>         type='ModuleOutputs',
            >>>         sources=['repeat_module1', 'repeat_module2'])]

            >>> ctx = RecorderManager(recorder_cfgs)
            >>> model = Model()
            >>> ctx.initialize(model)

            >>> ctx.recorders
            >>> {'ModuleOutputs': ModuleOutputsRecorder()}

            >>> with ctx:
            >>>     res = model(torch.ones(2))

            >>> ctx.get_record_data('ModuleOutputs', 'repeat_module1')
            >>> [(tensor([1., 1.]),)]
            >>> ctx.get_record_data('ModuleOutputs',
            >>>     'repeat_module1', data_index=1)
            >>> (tensor([1., 1.]),)
    """

    def __init__(self, recorders) -> None:

        self.recorders: Dict(str, BaseRecorder) = dict()
        for cfg in recorders:
            recorder_cfg = copy.deepcopy(cfg)
            record_type = cfg.type
            recorder_type = record_type + 'Recorder'
            assert record_type not in self.recorders, \
                f'{recorder_type} already exists. The recorders of the same \
                    type should be merged into one. Please check your config.'

            recorder_cfg.type = recorder_type
            self.recorders[record_type] = build_recorder(recorder_cfg)

    def get_record_data(self,
                        record_type: str,
                        source: str,
                        data_index: Optional[int] = None) -> Any:
        """Get data from corresponding recorder.

        Args:
            record_type (str): The type of recorder that the data belong to.
            source (str): The key of the data saved in corresponding recorder's
             ``data_buffer``.
            data_index (int, optional):  The index of record source data. The
                source data is a list saved in corresponding recorder's
                ``data_buffer``.

        Returns:
            Any: The type of the return value is undefined, and different
                source data may have different types.
        """
        recorder = self.recorders[record_type]
        data = recorder.get_record_data(source, data_index)
        return data

    def initialize(self, model):
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
