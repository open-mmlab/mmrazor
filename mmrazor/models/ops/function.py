# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module, functional


class InputResizer(Module):
    valid_interpolation_type = {
        'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area',
        'nearest-exact'
    }

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 interpolation_type: str = 'bicubic',
                 align_corners: bool = False,
                 scale_factor: Optional[Union[float, List[float]]] = None,
                 recompute_scale_factor: Optional[bool] = None) -> None:
        super().__init__()

        if size is not None:
            if len(size) != 2:
                raise ValueError('Length of size must be 2, '
                                 f'but got: {len(size)}')
        self._size = size
        if interpolation_type not in self.valid_interpolation_type:
            raise ValueError(
                'Expect `interpolation_type` be '
                f'one of {self.valid_interpolation_type}, but got: '
                f'{interpolation_type}')
        self._interpolation_type = interpolation_type
        self._scale_factor = scale_factor
        self._align_corners = align_corners
        self._recompute_scale_factor = recompute_scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._size is None:
            return x

        return functional.interpolate(
            input=x,
            size=self._size,
            mode=self._interpolation_type,
            scale_factor=self._scale_factor,
            align_corners=self._align_corners,
            recompute_scale_factor=self._recompute_scale_factor)
