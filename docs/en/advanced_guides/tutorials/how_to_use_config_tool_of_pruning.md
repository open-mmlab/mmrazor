# How to Use our Config Tool for Pruning

## How We Get MutableChannelUnits Automatically

Our pruning framework can automatically parse a model and get MutableChannelUnits.
It makes it easy to prune new models.

The parsing process is placed in ChannelUnitMutator.prepare_from_supernet. We first trace the model and get a graph, then we parse the graph and get MutableChannelUnits.

<p align='center'><img src="../../imgs/pruning/framework-graph.png" width=400 /></p>

## How to Get ChannelUnit Config Template

To make the configuration of ChannelUnit easy, we provide an interface to get the config template: ChannelMutator.config_template(). It returns a config dict. The config\['channel_unit_cfg'\]\['units\] store all parsed MutableChannelUnits.

```python
def config_template(self,
                    only_mutable_units=False,
                    with_unit_init_args=False,
                    with_channels=False):
    """Config template of the mutator.

    Args:
        only_mutable_units (bool, optional): Whether only return config of
            prunable units. It can omit unmutable MutableChannelUnits
            to decrease the length of the config. Defaults to False.
        with_unit_init_args (bool, optional): Whether return init_args of
            units. Let it be true, when you want to change the init
            args of units. Defaults to False.
        with_channels (bool, optional): Whether return channel info.
            The channel info can initialization the units without
            tracer. When you want to prune your model without a
            tracer next time, let it be true. Defaults to False.

    Example:
        dict(
            channel_unit_cfg = dict(
                # type of used MutableChannelUnit
                type ='XxxMutableChannelUnit',
                # default args for MutableChananelUnit
                default_args={},
                # config of units
                units = {
                    # config of a unit
                    "xxx_unit_name": {
                        'init_args':{}, # if with_unit_init_args
                        'channels':{} # if with_channels
                    },
                    ...
                }
            ),
            # config of tracer
            parse_cfg={}
        )


    About the detail of the config of each unit, please refer to
    MutableChannelUnit.config_template()
    """
```

Note the name of a unit is generated automatically according to their content, avoid to change the name in config.

Here, we give an example of getting a config template using code.

```python
from mmrazor.models.mutators import ChannelMutator
from torchvision.models import resnet34
model = resnet34()
# initialize a ChannelMutator object
mutator = ChannelMutator(
    channel_unit_cfg=dict(
        type='SequentialMutableChannelUnit',
        default_args=dict(choice_mode='ratio'),
        units={},
    ),
    parse_cfg=dict(
        type='BackwardTracer',
        loss_calculator=dict(type='ImageClassifierPseudoLoss')))
# init the ChannelMutator object with a model
mutator.prepare_from_supernet(model)
config=mutator.config_template(with_unit_init_args=True)
print(config)
# {
#     'type': 'ChannelMutator',
#     'channel_unit_cfg': {
#         'type': 'SequentialMutableChannelUnit',
#         'default_args': {
#             'choice_mode': 'ratio'
#         },
#         'units': {
#             'conv1_(0, 3)_3': {
#                 'init_args': {
#                     'num_channels': 3,
#                     'choice_mode': 'ratio',
#                     ...
#                 },
#                 'choice': 1.0
#             },
#            ...
#         }
#     },
#     'parse_cfg': {
#         'type': 'BackwardTracer',
#         'loss_calculator': {
#             'type': 'ImageClassifierPseudoLoss'
#         }
#     }
# }
```

Besides, it's also easy to initialize a new mutator using the config dict.

```python
# follow the code above
from mmrazor.registry import MODELS
mutator2=MODELS.build(config)
mutator2.prepare_from_supernet(resnet34())
```

To make your development more fluent, we provide a command tool to parse a model and return the config template.

```shell
$ python ./tools/get_channel_units.py -h

usage: get_channel_units.py [-h] [-c] [-i] [--choice] [-o OUTPUT_PATH] config

Get channel unit of a model.

positional arguments:
  config                config of the model

optional arguments:
  -h, --help            show this help message and exit
  -c, --with-channel    output with channel config
  -i, --with-init-args  output with init args
  --choice              output choices template. When this flag is activated, -c and -i will be ignored
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        the file path to store channel unit info
```

Take the algorithm Slimmable Network as an example.

```shell
python ./tools/get_channel_units.py ./configs/pruning/mmcls/autoslim/autoslim_mbv2_1.5x_slimmable_subnet_8xb256_in1k.py

# {
#     "type":"SlimmableChannelMutator",
#     "channel_unit_cfg":{
#         "type":"SlimmableChannelUnit",
#         "default_args":{},
#         "units":{
#             "backbone.conv1.conv_(0, 3)_3":{
#                 "choice":3
#             },
#             "backbone.conv1.conv_(0, 48)_48":{
#                 "choice":32
#             },
             ...
#         }
#     },
#     "parse_cfg":{
#         "type":"BackwardTracer",
#         "loss_calculator":{
#             "type":"ImageClassifierPseudoLoss"
#         }
#     }
# }
```

The '-i' flag will return the config with the initialization arguments.

```shell
python ./tools/get_channel_units.py -i ./configs/pruning/mmcls/autoslim/autoslim_mbv2_1.5x_slimmable_subnet_8xb256_in1k.py

# {
#     "type":"SlimmableChannelMutator",
#     "channel_unit_cfg":{
#         "type":"SlimmableChannelUnit",
#         "default_args":{},
#         "units":{
#             "backbone.conv1.conv_(0, 3)_3":{
#                 "init_args":{
#                     "num_channels":3,
#                     "divisor":1,
#                     "min_value":1,
#                     "min_ratio":0.9,
#                     "candidate_choices":[
#                         3
#                     ],
#                     "choice_mode":"number"
#                 },
#                 "choice":3
#             },
#             ...
#         }
#     },
#     "parse_cfg":{
#         "type":"BackwardTracer",
#         "loss_calculator":{
#             "type":"ImageClassifierPseudoLoss"
#         }
#     }
# }
```

With "--choice" flag, it will return the choice template, a dict which uses unit_name as key, and use the choice value as value.

```shell
python ./tools/get_channel_units.py -i ./configs/pruning/mmcls/autoslim/autoslim_mbv2_1.5x_slimmable_subnet_8xb256_in1k.py --choice

# {
#     "backbone.conv1.conv_(0, 48)_48":32,
#     "backbone.layer1.0.conv.1.conv_(0, 24)_24":16,
#     "backbone.layer2.0.conv.0.conv_(0, 144)_144":144,
#     "backbone.layer2.0.conv.2.conv_(0, 40)_40":24,
#     "backbone.layer2.1.conv.0.conv_(0, 240)_240":176,
#     "backbone.layer3.0.conv.0.conv_(0, 240)_240":192,
#     "backbone.layer3.0.conv.2.conv_(0, 48)_48":48,
#     "backbone.layer3.1.conv.0.conv_(0, 288)_288":240,
#     "backbone.layer3.2.conv.0.conv_(0, 288)_288":144,
#     "backbone.layer4.0.conv.0.conv_(0, 288)_288":264,
#     "backbone.layer4.0.conv.2.conv_(0, 96)_96":88,
#     "backbone.layer4.1.conv.0.conv_(0, 576)_576":288,
#     "backbone.layer4.2.conv.0.conv_(0, 576)_576":336,
#     "backbone.layer4.3.conv.0.conv_(0, 576)_576":432,
#     "backbone.layer5.0.conv.0.conv_(0, 576)_576":576,
#     "backbone.layer5.0.conv.2.conv_(0, 144)_144":144,
#     "backbone.layer5.1.conv.0.conv_(0, 864)_864":576,
#     "backbone.layer5.2.conv.0.conv_(0, 864)_864":648,
#     "backbone.layer6.0.conv.0.conv_(0, 864)_864":864,
#     "backbone.layer6.0.conv.2.conv_(0, 240)_240":240,
#     "backbone.layer6.1.conv.0.conv_(0, 1440)_1440":1440,
#     "backbone.layer6.2.conv.0.conv_(0, 1440)_1440":1440,
#     "backbone.layer7.0.conv.0.conv_(0, 1440)_1440":1440,
#     "backbone.layer7.0.conv.2.conv_(0, 480)_480":480,
#     "backbone.conv2.conv_(0, 1920)_1920":1920
# }
```
