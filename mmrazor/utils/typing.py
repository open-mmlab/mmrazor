# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Union

FixMutable = Dict[str, Any]
ValidFixMutable = Union[str, Path, FixMutable]

# RANDOM_SUBNET means the subnet sampled by one or more mutators. Usually used
# for supernet training or searching.

# `SingleMutatorRandomSubnet`` sampled by a mutator, its format is a dict, the
# keys of the dict are the group_id in the mutator‘s search groups, and the
# values ​​of the dict are the choices corresponding to all mutables in each
# search group.

# One search group may contains N mutables. More details of search groups see
# docs for :class:`mmrazor.models.mutators.OneShotModuleMutator`.
SingleMutatorRandomSubnet = Dict[int, Any]

# For some more complex algorithms, multiple mutators may be used, and the
# corresponding format will be a list
MultiMutatorsRandomSubnet = List[SingleMutatorRandomSubnet]

SupportRandomSubnet = Union[SingleMutatorRandomSubnet,
                            MultiMutatorsRandomSubnet]

Chosen = Union[str, float, List[str]]
ChosenMeta = Optional[Dict[str, Any]]


class DumpChosen(NamedTuple):
    chosen: Chosen
    meta: ChosenMeta = None


# DumpChosen = NamedTuple('DumpChosen', [('chosen', Chosen),
#                                        ('meta', ChosenMeta)])
