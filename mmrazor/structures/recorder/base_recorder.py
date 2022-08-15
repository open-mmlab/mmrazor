# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Any, List, Optional

from torch import nn


class BaseRecorder(metaclass=ABCMeta):
    """Base class for recorders.

    Recorder is a context manager used to record various intermediate results
    during the model forward. It can be used in distillation algorithm and
    can also be used to obtain some specific data for visual analysis.

    In MMRazor, there will be different types of recorders to obtain different
    types of intermediate results. They can be used in combination with the
    ``RecorderManager``.

    Note:
        The recorder will be lazily initialized in the ``RecorderManager`` by
        default. If you want to use the recorder without the
        ``RecorderManager``, you need to initialize it first.
    """

    def __init__(self, source: str) -> None:

        self._source = source
        # Intermediate results are recorded in dictionary format according
        # to the data source.
        # One data source may generate multiple records, which need to be
        # recorded through list.
        self._data_buffer: List = list()
        # Before using the recorder for the first time, it needs to be
        # initialized.
        self._initialized = False

    @property
    def source(self) -> str:
        """str: source of recorded data."""
        return self._source

    @property
    def data_buffer(self) -> List:
        """list: data buffer."""
        return self._data_buffer

    @abstractmethod
    def prepare_from_model(self, model: Optional[nn.Module] = None) -> None:
        """Make the intermediate results of the model can be record."""

    def initialize(self, model: Optional[nn.Module] = None) -> None:
        """Init the recorder.

        Args:
            model (nn.Module): The model which need to record intermediate
                results.
        """
        self.prepare_from_model(model)
        self._initialized = True

    def get_record_data(self,
                        record_idx: int = 0,
                        data_idx: Optional[int] = None) -> Any:
        """Get data from ``data_buffer``.

        Args:
            record_idx (int): The index of the record saved in
                ``data_buffer``. If a source is executed N times during
                forward, there will be N records in ``data_buffer``.
            data_index (int, optional):  The index of target data in
                a record. A record may be a tuple or a list, if data_idx is
                None, the whole list or tuple is returned. Defaults to None.

        Returns:
            Any: The type of the return value is undefined, and different
                source data may have different types.
        """
        assert record_idx < len(self._data_buffer), \
            'record_idx is illegal. The length of data_buffer is ' \
            f'{len(self._data_buffer)}, but record_idx is ' \
            f'{record_idx}.'

        record = self._data_buffer[record_idx]

        if data_idx is None:
            target_data = record
        else:
            if isinstance(record, (list, tuple)):
                assert data_idx < len(record), \
                    'data_idx is illegal. The length of record is ' \
                    f'{len(record)}, but data_idx is {data_idx}.'
                target_data = record[data_idx]
            else:
                raise TypeError('When data_idx is not None, record should be '
                                'a list or tuple instance, but got '
                                f'{type(record)}.')

        return target_data

    def reset_data_buffer(self) -> None:
        """Clear data in data_buffer."""

        self._data_buffer = list()

    def __enter__(self):
        """Enter the context manager."""

        assert self._initialized, \
            'The recorder will be initialized in the RecorderManager by '\
            'default. If you want to use the recorder without the '\
            'RecorderManager, you need to initialize it first.'

        self.reset_data_buffer()

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
