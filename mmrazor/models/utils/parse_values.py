# Copyright (c) OpenMMLab. All rights reserved.
from typing import List


def parse_values(candidate_lists: List[list]):
    """Parse a list with format `(min_range, max_range, step)`.

    NOTE: this method is required when customizing search space in configs.
    """

    def _range_to_list(input_range: List[int]) -> List[int]:
        assert len(input_range) == 3, (
            'The format should be `(min_range, max_range, step)` with dim=3, '
            f'but got dim={len(input_range)}.')
        start, end, step = input_range
        return list(range(start, end + 1, step))

    return [_range_to_list(i) for i in candidate_lists]
