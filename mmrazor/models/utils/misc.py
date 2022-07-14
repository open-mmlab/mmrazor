# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict


def add_prefix(inputs: Dict, prefix: str) -> Dict:
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:
        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs
