# Overview

## Why MMRazor

MMRazor is a model compression toolkit for model slimming, which includes 4 mainstream technologies:

- Neural Architecture Search (NAS)
- Pruning
- Knowledge Distillation (KD)
- Quantization

It is a part of the [OpenMMLab](https://openmmlab.com/) project. If you want to use it now, please refer to [Installation](https://mmrazor.readthedocs.io/en/main/get_started/installation.html).

### Major features:

- **Compatibility**

MMRazor can be easily applied to various projects in OpenMMLab, due to the similar architecture design of OpenMMLab as well as the decoupling of slimming algorithms and vision tasks.

- **Flexibility**

Different algorithms, e.g., NAS, pruning and KD, can be incorporated in a plug-n-play manner to build a more powerful system.

- **Convenience**

With better modular design, developers can implement new model compression algorithms with only a few codes, or even by simply modifying config files.

## Design and Implement

![overview - design and implement](https://user-images.githubusercontent.com/88702197/187396329-b5fedc96-c76b-49b7-af4e-83f1f0c27a57.jpg)

### Design

There are 3 layers (**Application** / **Algorithm** / **Component**) in overview design. MMRazor mainly includes both of **Component** and **Algorithm**, while **Application** consist of some OpenMMLab upstream repos, such as MMClassification,  MMDetection,  MMSegmentation and so on.

**Component** provides many useful functions for quickly implementing **Algorithm.** And thanks to OpenMMLab 's powerful and highly flexible config mode and registry mechanism, **Algorithm** can be conveniently applied to **Application.**

How to apply our lightweight algorithms to some upstream tasks? Please refer to the below.

### Implement

In OpenMMLab, implementing vision tasks commonly includes 3 parts (model / dataset / schedule). And just like that, implementing lightweight model also includes 3 parts (algorithm / dataset / schedule) in MMRazor.

`Algorithm` consist of `architecture` and `components`.

`Architecture` is similar to `model` of the upstream repos. You can chose to directly use the original `model` or customize the new `model` as your architecture according to different tasks. For example,  you can directly use ResNet-34 and ResNet-18 of MMClassification to implement some KD algorithms, but in NAS, you may need to customize a searchable model.

`Components` consist of various special functions for supporting different lightweight algorithms. They can be directly used in config because of  registered into MMEngine. Thus, you can pick some components you need to quickly implement your algorithm. For example, you may need `mutator` / `mutable` / `searchle backbone` if you want to implement a NAS algorithm, and you can pick from `distill loss` / `recorder` / `delivery` / `connector` if you need a KD algorithm.

Please refer to the next section for more details about **Implement**.

```{note}
The arg name of `algorithm` in config is **model** rather than **algorithm** in order to get better supports of MMCV and MMEngine.
```

## Key concepts

For better understanding and using MMRazor, it is highly recommended to read the following user documents according to your own needs.

**Global**

- [Algorithm](https://mmrazor.readthedocs.io/en/main/advanced_guides/algorithm.html)

**NAS & Pruning**

- [Mutator](https://mmrazor.readthedocs.io/en/main/advanced_guides/mutator.html)
- [Mutable](https://mmrazor.readthedocs.io/en/main/advanced_guides/mutable.html)

**KD**

- [Delivery](https://mmrazor.readthedocs.io/en/main/advanced_guides/delivery.html)
- [Recorder](https://mmrazor.readthedocs.io/en/main/advanced_guides/recorder.html)

## User guide

If you want to run mmrazor quickly, you can refer to as the follows.

- [Learn about Configs](https://mmrazor.readthedocs.io/en/main/user_guides/1_learn_about_config.html)
- [Train different types algorithms](https://mmrazor.readthedocs.io/en/main/user_guides/2_train_different_types_algorithms.html)
- [Train with different devices](https://mmrazor.readthedocs.io/en/main/user_guides/3_train_with_different_devices.html)
- [Test a model](https://mmrazor.readthedocs.io/en/main/user_guides/4_test_a_model.html)

## Tutorials

We provide the following general tutorials according to some typical requirements. If you want to further use MMRazor, you can refer to our source code and API Reference.

**Tutorial list**

- [Customize Architectures](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_architectures.html)
- [Customize NAS algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_nas_algorithms.html)
- [Customize Pruning algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_pruning_algorithms.html)
- [Customize KD algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_kd_algorithms.html)
- [Customize mixed algorithms](https://mmrazor.readthedocs.io/en/main/advanced_guides/customize_mixed_algorithms.html)
- [Apply existing algorithms to new tasks](https://mmrazor.readthedocs.io/en/main/advanced_guides/apply_existing_algorithms_to_new_tasks.html)

## F&Q

If you encounter some trouble using MMRazor, you can find whether your question has existed in [F&Q](https://mmrazor.readthedocs.io/en/main/notes/faq.html). If not existed, welcome to open a [Github issue](https://github.com/open-mmlab/mmrazor/issues) for getting support, we will reply it as soon.

## Get support and contribute back

MMRazor is maintained on the [MMRazor Github repository](https://github.com/open-mmlab/mmrazor). We collect feedback and new proposals/ideas on Github. You can:

- Open a [GitHub issue](https://github.com/open-mmlab/mmrazor/issues) for bugs and feature requests.
- Open a [pull request](https://github.com/open-mmlab/mmrazor/pulls) to contribute code (make sure to read the [contribution guide](https://mmrazor.readthedocs.io/en/main/notes/contribution_guide.html) before doing this).
