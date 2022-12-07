# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

from mmrazor.utils import print_log

warn_once = False


def make_divisible(value: int,
                   divisor: int,
                   min_value: Optional[int] = None,
                   min_ratio: float = 0.9) -> int:
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number
    """

    if min_value is None:
        min_value = divisor
    if min_value < divisor:
        global warn_once
        if warn_once is False:
            print_log((f'min_value=={min_value} should greater or equal to '
                       f'divisor=={divisor}, '
                       'so we make min_value equal divisor.'),
                      level='warning')
            warn_once = True

        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value
