# How to Use Config Tool of Pruning

## How We Get MutableChannelGroups Automatically

To extend new models easily, our pruning framework can parse a model and get MutableChannelGroups automatically. The parsing process is placed in ChannelGroupMutator.prepare_from_supernet.The process.

<p align='center'><img src="./images/../../images/framework-graph.png" width=400 /></p>

Please refer to (pruning graph) for details.

### How to Get ChannelGroup Config Template

To make the configuration of ChannelGroup easy, we provide an interface to get the config template: BaseChannelMutator.config_template(). It returns a dict to store the config. Each item in the dict represents a channel group.

```python
def config_template(self, with_init_args=False, with_channels=False) -> Dict:
    """
    Return the config template of this group. By default, the config template
    only includes a key 'choice'.

    Args:
        with_init_args (bool): if the config includes args for initialization.
        with_channels (bool): if the config includes info about channels.
    """
```

Here, we give an example of getting a config template using code.

```python
import json
from mmrazor.models.mutators import BaseChannelMutator
from mmengine.model import BaseModel
from mmrazor.registry import MODELS
from mmrazor.models.mutables import SequentialChannelGroup
from typing import Dict

@MODELS.register_module()
class MuiltArgsChannelGroup(SequentialChannelGroup):

    def __init__(self, num_channels: int, candidates=[]) -> None:
        super().__init__(num_channels)
        self.candidates = candidates

    def config_template(self, with_init_args=False, with_channels=False) -> Dict:
        config = super().config_template(with_init_args, with_channels)
        if with_init_args:
            init_args = config['init_args']
            init_args['candidates'] = self.candidates
        return config

class Model(BaseModel):
    ...

model = Model()
mutator = BaseChannelMutator(MuiltArgsChannelGroup)
mutator.prepare_from_supernet(model)
config = mutator.config_template(with_group_init_args=True)
print(json.dumps(config, indent=4))

# {
#     "feature.0_(0, 8)_out_1_in_1": {
#         "init_args": {
#             "num_channels": 8,
#             "candidates": []
#         },
#         "choice": 1.0
#     },
#     "feature.1_(0, 16)_out_1_in_1": {
#         "init_args": {
#             "num_channels": 16,
#             "candidates": []
#         },
#         "choice": 1.0
#     },
#     "head_(0, 1000)_out_1_in_0": {
#         "init_args": {
#             "num_channels": 1000,
#             "candidates": []
#         },
#         "choice": 1.0
#     }
# }
```

Besides, it's also easy to initialize a new mutator using the config dict.

```python
# follow the code above
mutator2 = BaseChannelMutator(
    dict(
        type='MuiltArgsChannelGroup',
        groups=config
    )
)
mutator2.prepare_from_supernet(Model())
```

To make your development more fluent, we provide a tool to parse a model and return ChannelGroup config template using terminal.

```shell
python ./tools/get_channel_groups.py -h

# usage: get_channel_groups.py [-h] [-c] [-o OUTPUT_PATH] config

# Get channel group of a model.

# positional arguments:
#   config                config of the model

# optional arguments:
#   -h, --help            show this help message and exit
#   -c, --with-channel    output with channel config
#   -o OUTPUT_PATH, --output-path OUTPUT_PATH
#                         the file path to store channel group info
```

Take the algorithm Slimmable Network  as the example.

```shell
python ./tools/get_channel_groups.py ./configs/pruning/mmcls/autoslim/autoslim_mbv2_1.5x_slimmable_subnet_8xb256_in1k.py

# {
#     "backbone.conv1.conv_(0, 48)_out_4_in_4":{
#         "init_args":{
#             "num_channels":48,
#             "candidates":[
#                 48
#             ]
#         },
#         "choice":48
#     },
#     ...
#     "head.fc_(0, 1000)_out_1_in_0":{
#         "init_args":{
#             "num_channels":1000,
#             "candidates":[
#                 1000
#             ]
#         },
#         "choice":1000
#     }
# }
```

With '-c' flag, it will channel information of the groups.

```shell
python ./tools/get_channel_groups.py -i ./configs/pruning/mmcls/autoslim/autoslim_mbv2_1.5x_slimmable_subnet_8xb256_in1k.py

{
#     "backbone.conv1.conv_(0, 48)_out_4_in_4":{# add tracer info
#         "init_args":{
#             "num_channels":48,
#             "candidates":[
#                 24,48
#             ]
#         },
#         "channels":{
#             "input_related":[
#                 {
#                     "name":"backbone.conv1.bn",
#                     "start":0,
#                     "end":48,
#                     "expand_ratio":1,
#                     "is_output_related":false
#                 },
#                 ...
#             ],
#             "output_related":[
#                 {
#                     "name":"backbone.conv1.conv",
#                     "start":0,
#                     "end":48,
#                     "expand_ratio":1,
#                     "is_output_related":true
#                 },
#                 ...
#             ]
#         },
#         "choice":48
#     },
#    ...
#     "head.fc_(0, 1000)_out_1_in_0":{
#         "init_args":{
#             "num_channels":1000,
#             "candidates":[
#                 1000
#             ]
#         },
#         "channels":{
#             "input_related":[],
#             "output_related":[
#                 {
#                     "name":"head.fc",
#                     "start":0,
#                     "end":1000,
#                     "expand_ratio":1,
#                     "is_output_related":true
#                 }
#             ]
#         },
#         "choice":1000
#     }
# }
```
