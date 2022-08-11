# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from mmrazor.models.architectures.dynamic_op import DynamicOP


def fix_dynamic_op(op: DynamicOP, fix_mutables: Optional[Dict] = None) -> None:
    for mutable_name in op.accpeted_mutables:
        mutable = getattr(op, mutable_name)
        if mutable is None:
            continue

        if fix_mutables is not None:
            chosen = fix_mutables[mutable_name]
        else:
            chosen = mutable.dump_chosen()

        mutable.fix_chosen(chosen)
