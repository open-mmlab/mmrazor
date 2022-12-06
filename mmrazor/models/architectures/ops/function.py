# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import Module, functional


class InputResizer(Module):
    valid_interpolation_type = {
        'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear', 'area',
        'nearest-exact'
    }

    def __init__(
            self,
            interpolation_type: str = 'bicubic',
            align_corners: bool = False,
            scale_factor: Optional[Union[float, List[float]]] = None) -> None:
        super().__init__()

        if interpolation_type not in self.valid_interpolation_type:
            raise ValueError(
                'Expect `interpolation_type` be '
                f'one of {self.valid_interpolation_type}, but got: '
                f'{interpolation_type}')
        self._interpolation_type = interpolation_type
        self._scale_factor = scale_factor
        self._align_corners = align_corners
        self._size = None

    def forward(self,
                x: torch.Tensor,
                size: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        size = size if size is not None else self._size

        return functional.interpolate(
            input=x,
            size=size,
            mode=self._interpolation_type,
            scale_factor=self._scale_factor,
            align_corners=self._align_corners)
