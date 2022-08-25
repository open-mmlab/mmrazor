# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractproperty

from ..base_mutable import BaseMutable
from ..derived_mutable import DerivedMethodMixin


class BaseMutableChannel(BaseMutable, DerivedMethodMixin):
    """A type of ``MUTABLES`` for single path supernet such as AutoSlim. In
    single path supernet, each module only has one choice invoked at the same
    time. A path is obtained by sampling all the available choices. It is the
    base class for one shot channel mutables.

    Args:
        num_channels (int): The raw number of channels.
        init_cfg (dict, optional): initialization configuration dict for
            ``BaseModule``. OpenMMLab has implement 5 initializer including
            `Constant`, `Xavier`, `Normal`, `Uniform`, `Kaiming`,
            and `Pretrained`.
    """

    def __init__(self, num_channels: int, **kwargs):
        super().__init__(**kwargs)
        self.name = ''
        self.num_channels = num_channels

    # choice

    @abstractproperty
    def current_choice(self):
        raise NotImplementedError()

    @current_choice.setter
    def current_choice(self):
        raise NotImplementedError()

    @abstractproperty
    def current_mask(self):
        raise NotImplementedError()

    @property
    def activated_channels(self):
        return (self.current_mask == 1).sum().item()

    # implementation of abstract methods

    def fix_chosen(self, chosen):
        """Fix mutable with subnet config. This operation would convert
        `unfixed` mode to `fixed` mode. The :attr:`is_fixed` will be set to
        True and only the selected operations can be retained.

        Args:
            chosen (str): The chosen key in ``MUTABLE``. Defaults to None.
        """
        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        self.is_fixed = True

    def dump_chosen(self):
        raise NotImplementedError()

    def num_choices(self) -> int:
        raise NotImplementedError()

    # others

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(name={self.name}, '
        repr_str += f'num_channels={self.num_channels}, '
        return repr_str
