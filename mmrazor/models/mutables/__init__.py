# Copyright (c) OpenMMLab. All rights reserved.
from .diff_mutable import DiffChoiceRoute, DiffOP, GumbelChoiceRoute
from .oneshot_mutable import OneShotMutable, OneShotOP

__all__ = [
    'OneShotOP', 'OneShotMutable', 'DiffOP', 'DiffChoiceRoute',
    'GumbelChoiceRoute'
]
