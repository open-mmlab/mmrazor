# Changelog of v1.x

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
