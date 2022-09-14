# Mutable

## Introduction

### What is Mutable

`Mutable` is one of basic function components  in NAS algorithms and some pruning algorithms, which makes supernet searchable by providing optional modules or parameters.

To understand it better, we take the mutable module as an example to explain as follows.

![1280X1280](https://user-images.githubusercontent.com/88702197/187410115-a5cd158c-aa0b-44ee-af96-7b14bb4972ad.PNG)

As shown in the figure above,  `Mutable` is a container that holds some candidate operations, thus it can sample candidates to constitute the subnet. `Supernet` usually consists of multiple `Mutable`, therefore, `Supernet` will be searchable with the help of `Mutable`. And all candidate operations in `Mutable` constitute the search space of `SuperNet`.

```{note}
If you want to know more about the relationship between Mutable and Mutator, please refer to [Mutator](https://mmrazor.readthedocs.io/en/dev-1.x/advanced_guides/mutator.html)
```

### Features

#### 1. Support module mutable

It is the common and basic function for NAS algorithms. We can use it to implement some classical one-shot NAS algorithms, such as [SPOS](https://arxiv.org/abs/1904.00420), [DetNAS ](https://arxiv.org/abs/1903.10979)and so on.

#### 2. Support parameter mutable

To implement more complicated and funny algorithms easier, we supported making some important parameters searchable, such as input channel, output channel, kernel size and so on.

What is more, we can implement **dynamic op** by using mutable parameters.

#### 3. Support deriving from mutable parameter

Because of the restriction of defined architecture, there may be correlations between some mutable parameters, **such as concat and expand ratio.**

```{note}
If conv3 = concat (conv1, conv2)

When out_channel (conv1) = 3, out_channel (conv2) = 4

Then in_channel (conv3) must be 7 rather than mutable.

So use derived mutable from conv1 and conv2 to generate in_channel (conv3)
```

With the help of derived mutable, we can meet these special requirements in some NAS algorithms and pruning algorithms. What is more, it can be used to deal with different granularity between search spaces.

### Supported mutables

![UML 图 (8)](https://user-images.githubusercontent.com/88702197/187410159-9ca6ba13-29ce-483a-aa3e-903bedf8a441.jpg)

As shown in the figure above.

- **White blocks** stand the basic classes, which include `BaseMutable` and `DerivedMethodMixin`. `BaseMutable` is the base class for all mutables, which defines required properties and abstracmethods. `DerivedMethodMixin` is a mixin class to provide mutable parameters with some useful methods to derive mutable.

- **Gray blocks** stand different types of base mutables.

  ```{note}
  Because there are correlations between channels of some layers, we divide mutable parameters into `MutableChannel` and `MutableValue`, so you can also think `MutableChannel` is a special `MutableValue`.
  ```

  For supporting module and parameters mutable, we provide `MutableModule`, `MutableChannel` and `MutableValue` these base classes to implement required basic functions. And we also add `OneshotMutableModule` and `DiffMutableModule` two types  based on `MutableModule` to meet different types of algorithms' requirements.

  For supporting deriving from mutable parameters, we make `MutableChannel` and `MutableValue` inherit from `BaseMutable` and `DerivedMethodMixin`, thus they can get derived functions provided by `DerivedMethodMixin`.

- **Red blocks** and **green blocks** stand registered classes for implementing some specific algorithms, which means that you can use them directly in configs. If they do not meet your requirements, you can also customize your mutable based on our base classes. If you are interested in their realization, please refer to their docstring.

## How to use existing mutables to configure searchable backbones

We will use `OneShotMutableOP` to build a `SearchableShuffleNetV2` backbone as follows.

1. Configure needed mutables

```Python
# we only use OneShotMutableOP, then take 4 ShuffleOP as its candidates.
_STAGE_MUTABLE = dict(
    _scope_='mmrazor',
    type='OneShotMutableOP',
    candidates=dict(
        shuffle_3x3=dict(type='ShuffleBlock', kernel_size=3),
        shuffle_5x5=dict(type='ShuffleBlock', kernel_size=5),
        shuffle_7x7=dict(type='ShuffleBlock', kernel_size=7),
        shuffle_xception=dict(type='ShuffleXception')))
```

2. Configure the `arch_setting` of `SearchableShuffleNetV2`

```Python
# Use the _STAGE_MUTABLE in various stages.
arch_setting = [
    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: channel, num_blocks, mutable_cfg.
    [64, 4, _STAGE_MUTABLE],
    [160, 4, _STAGE_MUTABLE],
    [320, 8, _STAGE_MUTABLE],
    [640, 4, _STAGE_MUTABLE]
]
```

3. Configure searchable backbone.

```Python
nas_backbone = dict(
    _scope_='mmrazor',
    type='SearchableShuffleNetV2',
    widen_factor=1.0,
    arch_setting=arch_setting)
```

Then you can use it in your architecture. If existing mutables do not meet your needs, you can also customize your needed mutable.

## How to customize your mutable

### About base mutable

Before customizing mutables, we need to know what some base mutables do.

**BaseMutable**

In order to implement the searchable mechanism, mutables need to own some base functions, such as changing status from mutable to fixed, recording the current status and current choice and so on. So in `BaseMutable`, these relevant abstract methods and properties will be defined as follows.

```Python
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABC, abstractmethod
from typing import Dict, Generic, Optional, TypeVar

from mmengine.model import BaseModule

CHOICE_TYPE = TypeVar('CHOICE_TYPE')
CHOSEN_TYPE = TypeVar('CHOSEN_TYPE')

class BaseMutable(BaseModule, ABC, Generic[CHOICE_TYPE, CHOSEN_TYPE]):

    def __init__(self,
                 alias: Optional[str] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.alias = alias
        self._is_fixed = False
        self._current_choice: Optional[CHOICE_TYPE] = None

    @property
    def current_choice(self) -> Optional[CHOICE_TYPE]:
        return self._current_choice

    @current_choice.setter
    def current_choice(self, choice: Optional[CHOICE_TYPE]) -> None:
        self._current_choice = choice

    @property
    def is_fixed(self) -> bool:
        return self._is_fixed

    @is_fixed.setter
    def is_fixed(self, is_fixed: bool) -> None:
        ......
        self._is_fixed = is_fixed

    @abstractmethod
    def fix_chosen(self, chosen: CHOSEN_TYPE) -> None:
       pass

    @abstractmethod
    def dump_chosen(self) -> CHOSEN_TYPE:
        pass

    @property
    @abstractmethod
    def num_choices(self) -> int:
        pass
```

**MutableModule**

Inherited from `BaseModule`, `MutableModule` not only owns its basic functions, but also needs some specialized functions to implement module mutable, such as getting all choices, executing forward computation.

```Python
# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from typing import Any, Dict, List, Optional

from ..base_mutable import CHOICE_TYPE, CHOSEN_TYPE, BaseMutable

class MutableModule(BaseMutable[CHOICE_TYPE, CHOSEN_TYPE]):

    def __init__(self,
                 module_kwargs: Optional[Dict[str, Dict]] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.module_kwargs = module_kwargs

    @property
    @abstractmethod
    def choices(self) -> List[CHOICE_TYPE]:
        """list: all choices.  All subclasses must implement this method."""

    @abstractmethod
    def forward(self, x: Any) -> Any:
        """Forward computation."""

    @property
    def num_choices(self) -> int:
        """Number of choices."""
        return len(self.choices)
```

If you want to know more about other types mutables, please refer to their docstring.

### Steps of customizing mutables

There are 4 steps to implement a custom mutable.

1. Registry a new mutable

2. Implement abstract methods.

3. Implement other methods.

4. Import the class

Then you can use your customized mutable in configs as in the previous chapter.

Let's use `OneShotMutableOP` as an example for customizing mutable.

#### 1. Registry a new mutable

First, you need to determine which type mutable to implement. Thus, you can implement your mutable faster by inheriting from correlative base mutable.

Then create a new file `mmrazor/models/mutables/mutable_module/one_shot_mutable_module`, class `OneShotMutableOP` inherits from `OneShotMutableModule`.

```Python
# Copyright (c) OpenMMLab. All rights reserved.
import random
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch.nn as nn
from torch import Tensor

from mmrazor.registry import MODELS
from ..base_mutable import CHOICE_TYPE, CHOSEN_TYPE
from .mutable_module import MutableModule

@MODELS.register_module()
class OneShotMutableOP(OneShotMutableModule[str, str]):
    ...
```

#### 2. Implement abstract methods

##### 2.1 Basic abstract methods

These basic abstract methods are mainly from `BaseMutable` and `MutableModule`, such as `fix_chosen`, `dump_chosen`, `choices` and `num_choices`.

```Python
@MODELS.register_module()
class OneShotMutableOP(OneShotMutableModule[str, str]):
    ......

    def fix_chosen(self, chosen: str) -> None:
        """Fix mutable with subnet config. This operation would convert
        `unfixed` mode to `fixed` mode. The :attr:`is_fixed` will be set to
        True and only the selected operations can be retained.
        Args:
            chosen (str): the chosen key in ``MUTABLE``. Defaults to None.
        """
        if self.is_fixed:
            raise AttributeError(
                'The mode of current MUTABLE is `fixed`. '
                'Please do not call `fix_chosen` function again.')

        for c in self.choices:
            if c != chosen:
                self._candidates.pop(c)

        self._chosen = chosen
        self.is_fixed = True

    def dump_chosen(self) -> str:
        assert self.current_choice is not None

        return self.current_choice

    @property
    def choices(self) -> List[str]:
        """list: all choices. """
        return list(self._candidates.keys())

    @property
    def num_choices(self):
        return len(self.choices)
```

##### 2.2 Specified abstract methods

In `OneShotMutableModule`, sample and forward these required abstract methods are defined, such as `sample_choice`, `forward_choice`, `forward_fixed`, `forward_all`. So we need to implement these abstract methods.

```Python
@MODELS.register_module()
class OneShotMutableOP(OneShotMutableModule[str, str]):
    ......

    def sample_choice(self) -> str:
        """uniform sampling."""
        return np.random.choice(self.choices, 1)[0]

    def forward_fixed(self, x: Any) -> Tensor:
        """Forward with the `fixed` mutable.
        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        Returns:
            Tensor: the result of forward the fixed operation.
        """
        return self._candidates[self._chosen](x)

    def forward_choice(self, x: Any, choice: str) -> Tensor:
        """Forward with the `unfixed` mutable and current choice is not None.
        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
            choice (str): the chosen key in ``OneShotMutableOP``.
        Returns:
            Tensor: the result of forward the ``choice`` operation.
        """
        assert isinstance(choice, str) and choice in self.choices
        return self._candidates[choice](x)

    def forward_all(self, x: Any) -> Tensor:
        """Forward all choices. Used to calculate FLOPs.
        Args:
            x (Any): x could be a Torch.tensor or a tuple of
                Torch.tensor, containing input data for forward computation.
        Returns:
            Tensor: the result of forward all of the ``choice`` operation.
        """
        outputs = list()
        for op in self._candidates.values():
            outputs.append(op(x))
        return sum(outputs)
```

#### 3. Implement other methods

After finishing some required methods, we need to add some special methods, such as `_build_ops`, because it is needed in building candidates for sampling.

```Python
@MODELS.register_module()
class OneShotMutableOP(OneShotMutableModule[str, str]):
    ......

    @staticmethod
    def _build_ops(
            candidates: Union[Dict[str, Dict], nn.ModuleDict],
            module_kwargs: Optional[Dict[str, Dict]] = None) -> nn.ModuleDict:
        """Build candidate operations based on choice configures.
        Args:
            candidates (dict[str, dict] | :obj:`nn.ModuleDict`): the configs
                for the candidate operations or nn.ModuleDict.
            module_kwargs (dict[str, dict], optional): Module initialization
                named arguments.
        Returns:
            ModuleDict (dict[str, Any], optional):  the key of ``ops`` is
                the name of each choice in configs and the value of ``ops``
                is the corresponding candidate operation.
        """
        if isinstance(candidates, nn.ModuleDict):
            return candidates

        ops = nn.ModuleDict()
        for name, op_cfg in candidates.items():
            assert name not in ops
            if module_kwargs is not None:
                op_cfg.update(module_kwargs)
            ops[name] = MODELS.build(op_cfg)
        return ops
```

#### 4. Import the class

You can either add the following line to `mmrazor/models/mutables/mutable_module/__init__.py`

```Python
from .one_shot_mutable_module import OneShotMutableModule

__all__ = ['OneShotMutableModule']
```

or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.mutables.mutable_module.one_shot_mutable_module'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

Customize `OneShotMutableOP` is over, then you can use it directly in your algorithm.
