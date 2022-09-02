# Recorder

## Introduction of Recorder

`Recorder` is a context manager used to record various intermediate results during the model forward. It can help `Delivery` finish data delivering by recording source data in some distillation algorithms. And it can also be used to obtain some specific data for visual analysis or other functions you want.

To adapt to more requirements, we implement multiple types of recorders to obtain different types of intermediate results in MMRazor. What is more, they can be used in combination with the `RecorderManager`.

In general, `Recorder` will help us expand more functions in implementing algorithms by recording various intermediate results.

## Usage of Recorder

Currently, we support five `Recorder`, as shown in the following table

| Recorder name           | Description                                 |
| ----------------------- | ------------------------------------------- |
| FunctionOutputsRecorder | Record output results of some functions     |
| MethodOutputsRecorder   | Record output results of some methods       |
| ModuleInputsRecorder    | Record input results of nn.Module           |
| ModuleOutputsRecorder   | Record output results of nn.Module          |
| ParameterRecorder       | Record intermediate parameters of nn.Module |

All of the recorders inherit from `BaseRecorder`. And these recorders can be managed by `RecorderManager` or just be used on their own.

Their relationship is shown below.

![UML 图 (10)](https://user-images.githubusercontent.com/88702197/187415394-926daba3-1d78-4f7e-b20a-7f9ff1e1582d.jpg)

### FunctionOutputsRecorder

`FunctionOutputsRecorder` is used to record the output results of intermediate **function**.

```{note}
When instantiating `FunctionOutputsRecorder`, you need to pass `source` argument, which requires extra attention. For example,
`anchor_inside_flags` is a function in mmdetection to check whether the
anchors are inside the border. This function is in
`mmdet/core/anchor/utils.py` and used in
`mmdet/models/dense_heads/anchor_head`. Then the `source` argument should be
`mmdet.models.dense_heads.anchor_head.anchor_inside_flags` but not
`mmdet.core.anchor.utils.anchor_inside_flags`.
```

#### Example

Suppose there is a toy function named `toy_func` in toy_module.py.

```Python
import random
from typing import List
from mmrazor.structures import FunctionOutputsRecorder

def toy_func() -> int:
    return random.randint(0, 1000000)

# instantiate with specifying used path
r1 = FunctionOutputsRecorder('toy_module.toy_func')

# initialize is to make specified module can be recorded by
# registering customized forward hook.
r1.initialize()
with r1:
    out1 = toy_module.toy_func()
    out2 = toy_module.toy_func()
    out3 = toy_module.toy_func()

# check recorded data
print(r1.data_buffer)
```

Out:

```Python
[75486, 641059, 119729]
```

Test Correctness of recorded results

```Python
data_buffer = r1.data_buffer
print(data_buffer[0] == out1 and data_buffer[1] == out2 and data_buffer[2] == out3)
```

Out:

```Python
True
```

To get the specific recorded data with `get_record_data`

```Python
print(r1.get_record_data(record_idx=2))
```

Out:

```Python
119729
```

### MethodOutputsRecorder

`MethodOutputsRecorder` is used to record the output results of intermediate **method**.

#### Example

Suppose there is a toy class `Toy` and it has a toy method `toy_func` in toy_module.py.

```Python
import random
from mmrazor.core import MethodOutputsRecorder

class Toy():
    def toy_func(self):
        return random.randint(0, 1000000)

toy = Toy()

# instantiate with specifying used path
r1 = MethodOutputsRecorder('toy_module.Toy.toy_func')
# initialize is to make specified module can be recorded by
# registering customized forward hook.
r1.initialize()

with r1:
    out1 = toy.toy_func()
    out2 = toy.toy_func()
    out3 = toy.toy_func()

# check recorded data
print(r1.data_buffer)
```

Out:

```Python
[217832, 353057, 387699]
```

Test Correctness of recorded results

```Python
data_buffer = r1.data_buffer
print(data_buffer[0] == out1 and data_buffer[1] == out2 and data_buffer[2] == out3)
```

Out:

```Python
True
```

To get the specific recorded data with `get_record_data`

```Python
print(r1.get_record_data(record_idx=2))
```

Out:

```Python
387699
```

### ModuleOutputsRecorder and ModuleInputsRecorder

`ModuleOutputsRecorder`'s usage is similar with `ModuleInputsRecorder`'s, so we will take the former as an example to introduce their usage.

#### Example

```{note}
> Different `MethodOutputsRecorder` and `FunctionOutputsRecorder`, `ModuleOutputsRecorder` and `ModuleInputsRecorder` are instantiated with module name rather than used path, and executing `initialize` need arg: `model`. Thus, they can know actually the module needs to be recorded.
```

Suppose there is a toy Module `ToyModule` in toy_module.py.

```Python
import torch
from torch import nn
from mmrazor.core import ModuleOutputsRecorder

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1(x + 1)
        return self.conv2(x1 + x2)

model = ToyModel()
# instantiate with specifying module name.
r1 = ModuleOutputsRecorder('conv1')

# initialize is to make specified module can be recorded by
# registering customized forward hook.
r1.initialize(model)

x = torch.randn(1, 1, 1, 1)
with r1:
    out = model(x)

print(r1.data_buffer)
```

Out:

```Python
[tensor([[[[0.0820]]]], grad_fn=<ThnnConv2DBackward0>), tensor([[[[-0.0894]]]], grad_fn=<ThnnConv2DBackward0>)]
```

Test Correctness of recorded results

```Python
print(torch.equal(r1.data_buffer[0], model.conv1(x)))
print(torch.equal(r1.data_buffer[1], model.conv1(x + 1)))
```

Out:

```Python
True
True
```

### ParameterRecorder

`ParameterRecorder` is used to record the intermediate parameter of `nn.Module`. Its usage is similar to `ModuleOutputsRecorder`'s and `ModuleInputsRecorder`'s, but it instantiates with parameter name instead of module name.

#### Example

Suppose there is a toy Module `ToyModule` in toy_module.py.

```Python
from torch import nn
import torch
from mmrazor.core import ModuleOutputsRecorder

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.toy_conv = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.toy_conv(x)

model = ToyModel()
# instantiate with specifying parameter name.
r1 = ParameterRecorder('toy_conv.weight')
# initialize is to make specified module can be recorded by
# registering customized forward hook.
r1.initialize(model)

print(r1.data_buffer)
```

Out:

```Python
[Parameter containing: tensor([[[[0.2971]]]], requires_grad=True)]
```

Test Correctness of recorded results

```Python
print(torch.equal(r1.data_buffer[0], model.toy_conv.weight))
```

Out:

```Python
True
```

### RecorderManager

`RecorderManager` is actually context manager, which can be used to manage various types of recorders.

With the help of `RecorderManager`,  we can manage several different recorders with as little code as possible, which reduces the possibility of errors.

#### Example

Suppose there is a toy class `Toy` owned has a toy method `toy_func` in toy_module.py.

```Python
import random
from torch import nn
from mmrazor.core import RecorderManager

class Toy():
    def toy_func(self):
        return random.randint(0, 1000000)

class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.conv2 = nn.Conv2d(1, 1, 1)
        self.toy = Toy()

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.toy.toy_func()

# configure multi-recorders
conv1_rec = ConfigDict(type='ModuleOutputs', source='conv1')
conv2_rec = ConfigDict(type='ModuleOutputs', source='conv2')
func_rec = ConfigDict(type='MethodOutputs', source='toy_module.Toy.toy_func')
# instantiate RecorderManager with a dict that contains recorders' configs,
# you can customize their keys.
manager = RecorderManager(
    {'conv1_rec': conv1_rec,
     'conv2_rec': conv2_rec,
     'func_rec': func_rec})

model = ToyModel()
# initialize is to make specified module can be recorded by
# registering customized forward hook.
manager.initialize(model)

x = torch.rand(1, 1, 1, 1)
with manager:
    out = model(x)

conv2_out = manager.get_recorder('conv2_rec').get_record_data()
print(conv2_out)
```

Out:

```Python
tensor([[[[0.5543]]]], grad_fn=<ThnnConv2DBackward0>)
```

Display output of `toy_func`

```Python
func_out = manager.get_recorder('func_rec').get_record_data()
print(func_out)
```

Out:

```Python
313167
```
