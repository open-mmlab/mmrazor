# Mutator

## Introduction

### What is Mutator

**Mutator** is one of algorithm components, which provides some useful functions used for mutable management, such as sample choice, set choicet and so on.  With Mutator's help, you can implement some NAS or pruning algorithms quickly.

### What is the relationship between Mutator and Mutable

![1280X1280](https://user-images.githubusercontent.com/88702197/187410115-a5cd158c-aa0b-44ee-af96-7b14bb4972ad.PNG)

In a word, Mutator is the manager of Mutable. Each different type of mutable is commonly managed by their one correlative mutator, respectively.

As shown in the figure, Mutable is a component of supernet, therefore Mutator can implement some functions about subnet from supernet by handling Mutable.

### Supported mutators

In MMRazor, we have implemented some mutators, their relationship is as below.

![UML 图 (9)](https://user-images.githubusercontent.com/88702197/187413945-7e960973-d90b-4ac8-9e38-15095302ebb4.jpg)

`BaseMutator`: Base class for all mutators. It has appointed some abstract methods supported by all mutators.

`ModuleMuator`/ `ChannelMutator`: Two different types mutators are for handling mutable module and mutable channel respectively.

```{note}
Please refer to [Mutable](https://mmrazor.readthedocs.io/en/main/advanced_guides/mutable.html) for more details about different types of mutable.
```

`OneShotModuleMutator` / `DiffModuleMutator`: Inherit from `ModuleMuator`, they are for implementing different types algorithms, such as [SPOS](https://arxiv.org/abs/1904.00420), [Darts](https://arxiv.org/abs/1806.09055) and so on.

`OneShotChannelMutator` / `SlimmableChannelMutator`: Inherit from `ChannelMutator`, they are also for meeting the needs of different types algorithms, such as [AotuSlim](https://arxiv.org/abs/1903.11728).

## How to use existing mutators

You just use them directly in configs as below

```Python
supernet = dict(
    ...
    )

model = dict(
    type='mmrazor.SPOS',
    architecture=supernet,
    mutator=dict(type='mmrazor.OneShotModuleMutator'))
```

If existing mutators do not meet your needs, you can also customize your needed mutator.

## How to customize your mutator

All mutators need to implement at least two of the following interfaces

- `prepare_from_supernet()`

  - Make some necessary preparations according to the given supernet. These preparations may include, but are not limited to, grouping the search space, and initializing mutator with the parameters needed for itself.

- `search_groups`

  - Group of search space.

  - Note that **search groups** and **search space** are two different concepts. The latter defines what choices can be used for searching. The former groups the search space, and searchable blocks that are grouped into the same group will share the same search space and the same sample result.

  - ```Python
    # Example
    search_space = {op1, op2, op3, op4}
    search_group = {0: [op1, op2], 1: [op3, op4]}
    ```

There are 4 steps to implement a custom mutator.

1. Registry a new mutator

2. Implement abstract methods

3. Implement other methods

4. Import the class

Then you can use your customized mutator in configs as in the previous chapter.

Let's use `OneShotModuleMutator` as an example for customizing mutator.

### 1.Registry a new mutator

First, you need to determine which type mutator to implement. Thus, you can implement your mutator faster by inheriting from correlative base mutator.

Then create a new file `mmrazor/models/mutators/module_mutator/one_shot_module_mutator`, class `OneShotModuleMutator` inherits from `ModuleMutator`.

```Python
from mmrazor.registry import MODELS
from .module_mutator import ModuleMutator

@MODELS.register_module()
class OneShotModuleMutator(ModuleMutator):
    ...
```

### 2. Implement abstract methods

2.1. Rewrite the `mutable_class_type` property

```Python
@MODELS.register_module()
class OneShotModuleMutator(ModuleMutator):

     @property
    def mutable_class_type(self):
        """One-shot mutable class type.
        Returns:
            Type[OneShotMutableModule]: Class type of one-shot mutable.
        """
        return OneShotMutableModule
```

2.2. Rewrite `search_groups` and `prepare_from_supernet()`

As the `prepare_from_supernet()` method and the `search_groups` property are already implemented in the `ModuleMutator` and we don't need to add our own logic, the second step is already over.

If you need to implement them by yourself, you can refer to these as follows.

2.3. **Understand** **`search_groups`** **(optional)**

Let's take an example to see what default `search_groups` do.

```Python
from mmrazor.models import OneShotModuleMutator, OneShotMutableModule

class SearchableModel(nn.Module):
    def __init__(self, one_shot_op_cfg):
        # assume `OneShotMutableModule` contains 4 choices:
        # choice1, choice2, choice3 and choice4
        self.choice_block1 = OneShotMutableModule(**one_shot_op_cfg)
        self.choice_block2 = OneShotMutableModule(**one_shot_op_cfg)
        self.choice_block3 = OneShotMutableModule(**one_shot_op_cfg)

    def forward(self, x: Tensor) -> Tensor:
        x = self.choice_block1(x)
        x = self.choice_block2(x)
        x = self.choice_block3(x)

        return x

supernet = SearchableModel(one_shot_op_cfg)
mutator1 = OneShotModuleMutator()
# build mutator1 from supernet.
mutator1.prepare_from_supernet(supernet)
>>> mutator1.search_groups.keys()
dict_keys([0, 1, 2])
```

In this case, each `OneShotMutableModule` will be divided into a group. Thus, the search groups have 3 groups.

If you want to custom group according to your requirement, you can implement it by passing the arg `custom_group`.

```Python
custom_group = [
    ['op1', 'op2'],
    ['op3']
]
mutator2 = OneShotMutator(custom_group)
mutator2.prepare_from_supernet(supernet)
```

Then `choice_block1` and `choice_block2` will share the same search space and the same sample result, and `choice_block3` will have its own independent search space. Thus, the search groups have only 2 groups.

```Python
>>> mutator2.search_groups.keys()
dict_keys([0, 1])
```

### 3. Implement other methods

After finishing some required methods, we need to add some special methods, such as `sample_choices` and `set_choices`.

```Python
from typing import Any, Dict

from mmrazor.registry import MODELS
from ...mutables import OneShotMutableModule
from .module_mutator import ModuleMutator

@MODELS.register_module()
class OneShotModuleMutator(ModuleMutator):

    def sample_choices(self) -> Dict[int, Any]:
        """Sampling by search groups.
        The sampling result of the first mutable of each group is the sampling
        result of this group.
        Returns:
            Dict[int, Any]: Random choices dict.
        """
        random_choices = dict()
        for group_id, modules in self.search_groups.items():
            random_choices[group_id] = modules[0].sample_choice()

        return random_choices

    def set_choices(self, choices: Dict[int, Any]) -> None:
        """Set mutables' current choice according to choices sample by
        :func:`sample_choices`.
        Args:
            choices (Dict[int, Any]): Choices dict. The key is group_id in
                search groups, and the value is the sampling results
                corresponding to this group.
        """
        for group_id, modules in self.search_groups.items():
            choice = choices[group_id]
            for module in modules:
                module.current_choice = choice

    @property
    def mutable_class_type(self):
        """One-shot mutable class type.
        Returns:
            Type[OneShotMutableModule]: Class type of one-shot mutable.
        """
        return OneShotMutableModule
```

### 4. Import the class

You can either add the following line to `mmrazor/models/mutators/module_mutator/__init__.py`

```Python
from .one_shot_module_mutator import OneShotModuleMutator

__all__ = ['OneShotModuleMutator']
```

or alternatively add

```Python
custom_imports = dict(
    imports=['mmrazor.models.mutators.module_mutator.one_shot_module_mutator'],
    allow_failed_imports=False)
```

to the config file to avoid modifying the original code.

Customize `OneShotModuleMutator` is over, then you can use it directly in your algorithm.
