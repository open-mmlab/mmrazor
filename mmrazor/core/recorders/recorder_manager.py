# Copyright (c) OpenMMLab. All rights reserved.
import copy

from ..builder import build_recorder


class RecorderManager():

    def __init__(self, recorders) -> None:

        self.recorders = dict()
        for cfg in recorders:
            recorder_cfg = copy.deepcopy(cfg)
            record_type = cfg.type
            recorder_type = record_type + 'Recorder'
            recorder_cfg.type = recorder_type
            self.recorders[recorder_type] = build_recorder(recorder_cfg)

    def get_record_data(self, record_type, source, data_index=None):
        recorder = self.recorders[record_type]
        data = recorder.get_record_data(source, data_index)
        return data

    def __enter__(self, model):

        for recorder in self.recorders.values():
            recorder.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):

        for recorder in self.recorders.values():
            recorder.__exit__()
