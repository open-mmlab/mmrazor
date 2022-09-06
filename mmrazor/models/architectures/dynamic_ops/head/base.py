# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod

from mmrazor.models.mutables.base_mutable import BaseMutable


class DynamicHead:

    @abstractmethod
    def connect_with_backbone(self,
                              backbone_output_mutable: BaseMutable) -> None:
        ...
