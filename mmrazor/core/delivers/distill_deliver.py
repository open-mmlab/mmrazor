# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class DistillDeliver(metaclass=ABCMeta):

    def __init__(self, max_keep_data, source='teacher', target='student'):
        assert source in ['student', 'teacher']
        assert target in ['student', 'teacher']
        self.current_mode = 'eval'
        self.data_queue = list()
        self.max_keep_data = max_keep_data
        self.source = source
        self.target = target
        self.delivering = False

    def convert_mode(self, mode):
        assert mode in ['student', 'teacher', 'eval']
        self.current_mode = mode

    @abstractmethod
    def deliver_wrapper(self):
        pass

    def __enter__(self):
        self.delivering = True

    def __exit__(self, exc_type, exc_value, traceback):
        self.delivering = False
        self.convert_mode('eval')
