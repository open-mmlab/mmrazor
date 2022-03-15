# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseRecorder(metaclass=ABCMeta):

    def __init__(self):
        self.recording = False
        self.data_buffer = dict()
        self.initialized = False

    @abstractmethod
    def reset_data_buffer(self):
        pass

    @abstractmethod
    def prepare_from_model(self, model):
        pass

    def initialize(self, model):
        self.prepare_from_model(model)
        self.initialized = True

    def get_record_data(self, source, data_index=None):
        data = self.data_buffer[source]
        if data_index:
            assert isinstance(data_index, int)
            assert data_index < len(data)
            return data[data_index]
        else:
            return data

    def __enter__(self):
        assert self.initialized
        self.recording = True
        self.reset_data_buffer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.recording = False
