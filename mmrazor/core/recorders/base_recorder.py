# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional

from torch import nn


class BaseRecorder(metaclass=ABCMeta):
    """Base class for recorders. Recorder is a context manager used to record
    various intermediate results during the model forward. It can be used in
    distiller and can also be used to obtain some specific data for visual
    analysis.

    In MMRazor, there will be different types of recorders to obtain different
    types of intermediate results. They can be used in combination with the
    ``RecorderManager``.

    Note:
        The recorder will be lazily initialized in the ``RecorderManager`` by
        default. If you want to use the recorder without the
        ``RecorderManager``, you need to initialize it first.
    """

    def __init__(self):
        # Control whether to record.
        # The `recording` will be set to True when enter the context manager,
        # and will be reset to  False when exit the context manager.
        self.recording = False
        # A recorder will record multiple intermediate results of this type,
        # which are recorded in dictionary format according to the data source.
        self.data_buffer: Dict(str, list) = dict()
        # All recorders will lazy init in `RecorderManager`.
        self.initialized = False

    @abstractmethod
    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """Make the intermediate results of the model can be record."""
        pass

    def initialize(self, model: Optional[nn.Module] = None) -> None:
        """Init the recorder.

        Args:
            model (nn.Module): The model which need to record intermediate
                results.
        """
        self.prepare_from_model(model)
        self.initialized = True

    def get_record_data(self,
                        source: str,
                        data_index: Optional[int] = None) -> Any:
        """Get data from ``data_buffer``.

        Args:
            source (str): The key of the data saved in ``data_buffer``.
            data_index (int, optional):  The index of source data saved in
                ``data_buffer``.

        Returns:
            Any: The type of the return value is undefined, and different
                source data may have different types.
        """
        # self.data_buffer: Dict(str, List)
        data: List(Any) = self.data_buffer[source]

        if data_index is not None:
            assert isinstance(data_index, int), 'data_index must be int.'
            assert data_index < len(data), 'data_index is illegal.'
            return data[data_index]
        else:
            return data

    def reset_data_buffer(self) -> None:
        """Clear data in data_buffer."""
        for key in self.data_buffer.keys():
            self.data_buffer[key] = list()

    def __enter__(self):
        """Enter the context manager."""
        assert self.initialized, \
            'The recorder will be initialized in the RecorderManager by \
            default. If you want to use the recorder without the \
            RecorderManager, you need to initialize it first.'

        self.recording = True
        self.reset_data_buffer()

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.recording = False
