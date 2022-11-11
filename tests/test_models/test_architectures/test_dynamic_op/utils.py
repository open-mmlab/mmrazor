# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

from mmrazor.models.architectures.dynamic_ops import DynamicMixin
from mmrazor.utils.typing import DumpChosen


def fix_dynamic_op(op: DynamicMixin,
                   fix_mutables: Optional[Dict] = None) -> None:
    for name, mutable in op.mutable_attrs.items():

        if fix_mutables is not None:
            chosen = fix_mutables[f'mutable_attrs.{name}']
        else:
            chosen = mutable.dump_chosen()

        if not isinstance(chosen, DumpChosen):
            chosen = DumpChosen(**chosen)

        mutable.fix_chosen(chosen.chosen)
