# Copyright (c) OpenMMLab. All rights reserved.
import copy

from ..builder import build_deliver


class DistillDeliverManager():

    def __init__(self, deliveries) -> None:

        self.deliveries = dict()
        for cfg in deliveries:
            deliver_cfg = copy.deepcopy(cfg)
            deliver_type = cfg.type
            deliver_type = deliver_type + 'Deliver'
            deliver_cfg.type = deliver_type
            self.deliveries[deliver_type] = build_deliver(deliver_cfg)

    def convert_mode(self, mode):
        for deliver in self.deliveries.values():
            deliver.convert_mode(mode)

    def __enter__(self):

        for deliver in self.deliveries.values():
            deliver.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):

        for deliver in self.deliveries.values():
            deliver.__exit__(exc_type, exc_value, traceback)
