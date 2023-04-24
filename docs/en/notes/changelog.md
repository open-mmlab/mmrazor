# Changelog of v1.x

## v1.0.0 (24/04/2023)

We are excited to announce the first official release of MMRazor 1.0.

### Highlights

- MMRazor quantization is released, which has got through task models and model deployment. With its help, we can quantize and deploy pre-trained models in OpenMMLab to specified backend quickly.

### New Features & Improvements

#### NAS

- Update searchable model. (https://github.com/open-mmlab/mmrazor/pull/438)
- Update NasMutator to build search_space in NAS. (https://github.com/open-mmlab/mmrazor/pull/426)

#### Pruning

- Add a new pruning algorithm named GroupFisher. We support the full pipeline for GroupFisher, including pruning, finetuning and deployment.(https://github.com/open-mmlab/mmrazor/pull/459)

#### KD

- Support stopping distillation after a certain epoch. (https://github.com/open-mmlab/mmrazor/pull/455)
- Support distilling rtmdet with mmrazor, refer to here. (https://github.com/open-mmlab/mmyolo/pull/544)
- Add mask channel in MGD Loss. (https://github.com/open-mmlab/mmrazor/pull/461)

#### Quantization

- Support two quantization types: QAT and PTQ (https://github.com/open-mmlab/mmrazor/pull/513)
- Support various quantization bits. (https://github.com/open-mmlab/mmrazor/pull/513)
- Support various quantization methods, such as per_tensor / per_channel, symmetry / asymmetry and so on. (https://github.com/open-mmlab/mmrazor/pull/513)
- Support deploy quantized models to multiple backends, such as OpenVINO, TensorRT and so on. (https://github.com/open-mmlab/mmrazor/pull/513)
- Support applying quantization algorithms to multiple task repos directly, such as mmcls, mmdet and so on. (https://github.com/open-mmlab/mmrazor/pull/513)

### Bug Fixes

- Fix split in Darts config. (https://github.com/open-mmlab/mmrazor/pull/451)
- Fix a bug in Recorders. (https://github.com/open-mmlab/mmrazor/pull/446)
- Fix a bug when using get_channel_unit.py. (https://github.com/open-mmlab/mmrazor/pull/432)
- Fix a bug when deploying a pruned model to cuda. (https://github.com/open-mmlab/mmrazor/pull/495)

### Contributors

A total of 10 developers contributed to this release.
Thanks @415905716 @gaoyang07 @humu789 @LKJacky @HIT-cwh @aptsunny @cape-zck @vansin @twmht @wm901115nwpu

## v1.0.0rc2 (06/01/2023)

We are excited to announce the release of MMRazor 1.0.0rc2.

### New Features

#### NAS

- Add Performance Predictor: Support 4 performance predictors with 4 basic machine learning algorithms, which can be used to directly predict model accuracy without evaluation.(https://github.com/open-mmlab/mmrazor/pull/306)

- Support [Autoformer](https://arxiv.org/pdf/2107.00651.pdf), a one-shot architecture search algorithm dedicated to vision transformer search.(https://github.com/open-mmlab/mmrazor/pull/315 )

- Support [BigNAS](https://arxiv.org/pdf/2003.11142), a NAS algorithm which searches the following items in MobileNetV3 with the one-shot paradigm: kernel_sizes, out_channels, expand_ratios, block_depth and input sizes. (https://github.com/open-mmlab/mmrazor/pull/219 )

#### Pruning

- Support [DCFF](https://arxiv.org/abs/2107.06916), a filter channel pruning algorithm dedicated to efficient image classification.(https://github.com/open-mmlab/mmrazor/pull/295)

- We release a powerful tool to automatically analyze channel dependency, named ChannelAnalyzer. Here is an example as shown below.(https://github.com/open-mmlab/mmrazor/pull/371)

Now, ChannelAnalyzer supports most of CNN models in torchvision, mmcls, mmseg and mmdet. We will continue to support more models.

```python
from mmrazor.models.task_modules import ChannelAnalyzer
from mmengine.hub import get_model
import json

model = get_model('mmdet::retinanet/retinanet_r18_fpn_1x_coco.py')
unit_configs: dict = ChannelAnalyzer().analyze(model)
unit_config0 = list(unit_configs.values())[0]
print(json.dumps(unit_config0, indent=4))
# # short version of the config
# {
#     "channels": {
#         "input_related": [
#             {"name": "backbone.layer2.0.bn1"},
#             {“name": "backbone.layer2.0.conv2"}
#         ],
#         "output_related": [
#             {"name": "backbone.layer2.0.conv1"},
#             {"name": "backbone.layer2.0.bn1"}
#         ]
#     },
#}
```

#### KD

- Support [MGD](https://arxiv.org/abs/2205.01529), a detection distillation algorithm.(https://github.com/open-mmlab/mmrazor/pull/381)

### Bug Fixes

- Fix `FpnTeacherDistll` techer forward from `backbone + neck + head` to `backbone + neck`(#387 )
- Fix some expire configs and checkpoints(#373 #372 #422 )

### Ongoing Changes

We will release Quantization in next version(1.0.0rc3)!

### Contributors

A total of 11 developers contributed to this release: @wutongshenqiu @sunnyxiaohu @aptsunny @humu789 @TinyTigerPan @FreakieHuang @LKJacky @wilxy @gaoyang07 @spynccat @yivona08.

## v1.0.0rc1 (27/10/2022)

We are excited to announce the release of MMRazor 1.0.0rc1.

### Highlights

- **New Pruning Framework**：We have systematically refactored the Pruning module. The new Pruning module can more automatically resolve the dependencies between channels and cover more corner cases.

### New Features

#### Pruning

- A new pruning framework is released in this release. (#311, #313)
  It consists of five core modules, including Algorithm, `ChannelMutator`, `MutableChannelUnit`, `MutableChannel` and `DynamicOp`.

- MutableChannelUnit is introduced for the first time. Each MutableChannelUnit manages all channels with channel dependency.

  ```python
  from mmrazor.registry import MODELS

  ARCHITECTURE_CFG = dict(
      _scope_='mmcls',
      type='ImageClassifier',
      backbone=dict(type='MobileNetV2', widen_factor=1.5),
      neck=dict(type='GlobalAveragePooling'),
      head=dict(type='mmcls.LinearClsHead', num_classes=1000, in_channels=1920))
  model = MODELS.build(ARCHITECTURE_CFG)
  from mmrazor.models.mutators import ChannelMutator

  channel_mutator = ChannelMutator()
  channel_mutator.prepare_from_supernet(model)
  units = channel_mutator.mutable_units
  print(units[0])
  # SequentialMutableChannelUnit(
  #   name=backbone.conv1.conv_(0, 48)_48
  #   (output_related): ModuleList(
  #     (0): Channel(backbone.conv1.conv, index=(0, 48), is_output_channel=true, expand_ratio=1)
  #     (1): Channel(backbone.conv1.bn, index=(0, 48), is_output_channel=true, expand_ratio=1)
  #     (2): Channel(backbone.layer1.0.conv.0.conv, index=(0, 48), is_output_channel=true, expand_ratio=1)
  #     (3): Channel(backbone.layer1.0.conv.0.bn, index=(0, 48), is_output_channel=true, expand_ratio=1)
  #   )
  #   (input_related): ModuleList(
  #     (0): Channel(backbone.conv1.bn, index=(0, 48), is_output_channel=false, expand_ratio=1)
  #     (1): Channel(backbone.layer1.0.conv.0.conv, index=(0, 48), is_output_channel=false, expand_ratio=1)
  #     (2): Channel(backbone.layer1.0.conv.0.bn, index=(0, 48), is_output_channel=false, expand_ratio=1)
  #     (3): Channel(backbone.layer1.0.conv.1.conv, index=(0, 48), is_output_channel=false, expand_ratio=1)
  #   )
  #   (mutable_channel): SquentialMutableChannel(num_channels=48, activated_channels=48)
  # )
  ```

Our new pruning algorithm can help you develop pruning algorithm more fluently. Pelease refer to our documents [PruningUserGuide](<./docs/en/user_guides/../../pruning/%5Bpruning_user_guide.md%5D(http://pruning_user_guide.md/)>) for model detail.

#### Distillation

- Support [CRD](https://arxiv.org/abs/1910.10699), a distillation algorithm based on contrastive representation learning. (#281)

- Support [PKD](https://arxiv.org/abs/2207.02039), a distillation algorithm that can be used in `MMDetection` and `MMDetection3D`. #304

- Support [DEIT](https://arxiv.org/abs/2012.12877), a classic **Transformer** distillation algorithm.(#332)

- Add a more powerful baseline setting for [KD](https://arxiv.org/abs/1503.02531). (#305)

- Add `MethodInputsRecorder` and `FuncInputsRecorder` to record the input of a class method or a function.(#320)

#### NAS

- Support [DSNAS](https://arxiv.org/pdf/2002.09128.pdf), a nas algorithm that does not require retraining. (#226 )

#### Tools

- Support configurable immediate feature map visualization. (#293 )
  A useful tool is supported in this release to visualize the immediate features of a neural network. Please refer to our documents [VisualizationUserGuide](http://./docs/zh_cn/user_guides/visualization.md) for more details.

### Bug Fixes

- Fix the bug that `FunctionXXRecorder` and `FunctionXXDelivery` can not be pickled. (#320)

### Ongoing changes

- Quantization:  We are developing the basic interface of PTQ and QAT. RFC(Request for Comments) will be released soon.
- AutoSlim: AutoSlim is not yet available and is being refactored.
- Fx Pruning Tracer: Currently, the model topology can only be resolved through the backward tracer. In the future, both backward tracer and fx tracer will be supported.
- More Algorithms: BigNAS、AutoFormer、GreedyNAS and Resrep will be released in the next few versions.
- Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMRazor 1.x.

### Contributors

A total of 12 developers contributed to this release.
Thanks @FreakieHuang @gaoyang07 @HIT-cwh @humu789 @LKJacky @pppppM @pprp @spynccat @sunnyxiaohu @wilxy @kitecats @SheffieldCao

## v1.0.0rc0 (31/8/2022)

We are excited to announce the release of MMRazor 1.0.0rc0.
MMRazor 1.0.0rc0 is the first version of MMRazor 1.x, a part of the OpenMMLab 2.0 projects.
Built upon the new [training engine](https://github.com/open-mmlab/mmengine),
MMRazor 1.x simplified the interaction with other OpenMMLab repos, and upgraded the basic APIs of KD / Pruning / NAS.
It also provides a series of knowledge distillation algorithms.

### Highlights

- **New engines**. MMRazor 1.x is based on [MMEngine](https://github.com/open-mmlab/mmengine), which provides a general and powerful runner that allows more flexible customizations and significantly simplifies the entrypoints of high-level interfaces.

- **Unified interfaces**. As a part of the OpenMMLab 2.0 projects, MMRazor 1.x unifies and refactors the interfaces and internal logic of train, testing, datasets, models, evaluation, and visualization. All the OpenMMLab 2.0 projects share the same design in those interfaces and logic to allow the emergence of multi-task/modality algorithms.

- **More configurable KD**.  MMRazor 1.x add [Recorder](../advanced_guides/recorder.md) to get the data needed for KD more automatically，[Delivery ](../advanced_guides/delivery.md) to automatically pass the teacher's intermediate results to the student， and connector to handle feature dimension mismatches between teacher and student.

- **More kinds of KD algorithms**. Benefitting from the powerful APIs of KD， we have added several categories of KD algorithms, data-free distillation, self-distillation, and zero-shot distillation.

- **Unify the basic interface of NAS and Pruning**. We refactored [Mutable](../advanced_guides/mutable.md), adding mutable value and mutable channel. Both NAS and Pruning can be developed based on mutables.

- **More documentation and tutorials**. We add a bunch of documentation and tutorials to help users get started more smoothly. Read it [here](https://mmrazor.readthedocs.io/en/1.0.0rc0/).

### Breaking Changes

#### Training and testing

- MMRazor 1.x runs on PyTorch>=1.6. We have deprecated the support of PyTorch 1.5 to embrace the mixed precision training and other new features since PyTorch 1.6. Some models can still run on PyTorch 1.5, but the full functionality of MMRazor 1.x is not guaranteed.
- MMRazor 1.x uses Runner in [MMEngine](https://github.com/open-mmlab/mmengine) rather than that in MMCV. The new Runner implements and unifies the building logic of dataset, model, evaluation, and visualizer. Therefore, MMRazor 1.x no longer maintains the building logics of those modules in `mmdet.train.apis` and `tools/train.py`. Those code have been migrated into [MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py).
- The Runner in MMEngine also supports testing and validation. The testing scripts are also simplified, which has similar logic as that in training scripts to build the runner.

#### Configs

- The [Runner in MMEngine](https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/runner.py) uses a different config structures
- Config and model names

#### Components

- Algorithms
- Distillers
- Mutators
- Mutables
- Hooks

### Improvements

- Support mixed precision training of all the models. However, some models may got Nan results due to some numerical issues. We will update the documentation and list their results (accuracy of failure) of mixed precision training.

### Bug Fixes

- AutoSlim: Models of different sizes will no longer have the same size checkpoint

### New Features

- Support [Activation Boundaries Loss](https://arxiv.org/pdf/1811.03233.pdf)
- Support [Be Your Own Teacher](https://arxiv.org/abs/1905.08094)
- Support [Data-Free Learning of Student Networks](https://doi.org/10.1109/ICCV.2019.00361)
- Support [Data-Free Adversarial Distillation](https://arxiv.org/pdf/1912.11006.pdf)
- Support [Decoupled Knowledge Distillation](https://arxiv.org/pdf/2203.08679.pdf)
- Support [Factor Transfer](https://arxiv.org/abs/1802.04977)
- Support [FitNets](https://arxiv.org/abs/1412.6550)
- Support [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)
- Support [Overhaul](https://arxiv.org/abs/1904.01866)
- Support [Zero-shot Knowledge Transfer via Adversarial Belief Matching](https://arxiv.org/abs/1905.09768)

### Ongoing changes

- Quantization:  We are developing the basic interface of PTQ and QAT. RFC(Request for Comments) will be released soon.
- AutoSlim: AutoSlim is not yet available and is being refactored.
- Fx Pruning Tracer: Currently, the model topology can only be resolved through the backward tracer. In the future, both backward tracer and fx tracer will be supported.
- More Algorithms: BigNAS、AutoFormer、GreedyNAS and Resrep will be released in the next few versions.
- Documentation: we will add more design docs, tutorials, and migration guidance so that the community can deep dive into our new design, participate the future development, and smoothly migrate downstream libraries to MMRazor 1.x.

### Contributors

A total of 13 developers contributed to this release.
Thanks @FreakieHuang @gaoyang07 @HIT-cwh @humu789 @LKJacky @pppppM @pprp @spynccat @sunnyxiaohu @wilxy @wutongshenqiu @NickYangMin @Hiwyl
Special thanks to @Davidgzx for his contribution to the data-free distillation algorithms
