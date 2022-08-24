# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, Type, TypeVar

from mmengine.model import BaseModule
from torch.nn import Module

from ..mutables.base_mutable import BaseMutable

MUTABLE_TYPE = TypeVar('MUTABLE_TYPE', bound=BaseMutable)


class BaseMutator(ABC, BaseModule, Generic[MUTABLE_TYPE]):
    """The base class for mutator.

    Mutator is mainly used for subnet management, it usually provides functions
    such as sampling and setting of subnets.

    All subclasses should implement the following APIs:

    - ``prepare_from_supernet()``
    - ``search_space``

    Args:
        init_cfg (dict, optional): The config to control the initialization.
    """

    def __init__(self, init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

    @abstractmethod
    def prepare_from_supernet(self, supernet: Module) -> None:
        """Do some necessary preparations with supernet.

        Args:
            supernet (:obj:`torch.nn.Module`): The supernet to be searched
                in your algorithm.
        """

    @property
    @abstractmethod
    def search_groups(self) -> Dict:
        """Search group of the supernet.

        Note:
            Search group is different from search space. The key of search
            group is called ``group_id``, and the value is corresponding
            searchable modules. The searchable modules will have the same
            search space if they are in the same group.

        Returns:
            dict: Search group.
        """

    @property
    @abstractmethod
    def mutable_class_type(self) -> Type[MUTABLE_TYPE]:
        """Corresponding mutable class type."""
        pass
