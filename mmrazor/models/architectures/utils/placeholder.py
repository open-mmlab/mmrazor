# Copyright (c) OpenMMLab. All rights reserved.
from torch import nn


class Placeholder(nn.Module):
    """Used for build searchable network.

    Args:
        group (str): Placeholder_group, as group name in searchable network.
        space_id (str): It is one and only index for each ``Placeholder``.
        choices (dict): Consist of the registered ``OPS``, used to combine
            ``MUTABLES``, the ``Placeholder`` will be replace with the
            ``MUTABLES``.
        choice_args (dict): The configuration of ``OPS`` used in choices.
    """

    def __init__(self, group, space_id, choices=None, choice_args=None):
        super(Placeholder, self).__init__()
        self.placeholder_group = group
        self.placeholder_kwargs = dict(space_id=space_id)
        if choices is not None:
            self.placeholder_kwargs.update(dict(choices=choices))
        if choice_args is not None:
            self.placeholder_kwargs.update(dict(choice_args=choice_args))
