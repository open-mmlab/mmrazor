<div align="center">
  <img src="resources/mmrazor-logo.png" width="600"/>
</div>
<br />

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mmrazor)](https://pypi.org/project/mmrazor/)
[![PyPI](https://img.shields.io/pypi/v/mmrazor)](https://pypi.org/project/mmrazor)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmrazor.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmrazor/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrazor/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrazor/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrazor)
[![license](https://img.shields.io/github/license/open-mmlab/mmrazor.svg)](https://github.com/open-mmlab/mmrazor/blob/master/LICENSE)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmrazor.svg)](https://github.com/open-mmlab/mmrazor/issues)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmrazor.svg)](https://github.com/open-mmlab/mmrazor/issues)

Documentation: https://mmrazor.readthedocs.io/

English | [简体中文](/README_zh-CN.md)

## Introduction

MMRazor is an open source model compression toolbox based on PyTorch. It is
a part of the [OpenMMLab](https://openmmlab.com/) project.


<details open><summary>Major features</summary>

- **All in one**

  MMRazor includes Neural Architecture Search（NAS), Knowledge Distillation（KD), Pruning, and Quantization（coming soon).
  And it supports the combination of different types of algorithms.

- **General CV Model Compression ToolBox**

  Thanks to OpenMMLab, the algorithms in MMRazor can be quickly applied to different CV tasks, making the development of model compression algorithms once and for all.

- **Decouple model and compression algorithm**

  MMRazor has a variety of built-in automation mechanisms, allowing developers to implement model compression algorithms without modifying the raw model code, such as:
  - OP can be modified code-free.
  - Feature maps in the middle layers can be obtained code-free.
  - Automatically obtain and analyze the connection between nn.Module.
  - And More ...

- **Flexible and Modular Design**

  We decompose the model compression algorithms into different components, making it much easier and more flexible to build a new algorithm by combining different components.

</details>

Below is an overview of MMRazor's design and implementation, please refer to [tutorials](/docs/en/tutorials/overview.md) for more details.
<div align="center">
  <img src="resources/design_and_implement.png" style="zoom:100%"/>
</div>
<br />

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.1.0 was released in 12/23/2021.

## Benchmark and model zoo

Results and models are available in the [model zoo](/docs/en/model_zoo.md).

Supported algorithms:

- <details open><summary>Neural Architecture Search</summary>

  - [x] [DARTS(ICLR'2019)](/configs/nas/darts)

  - [x] [DetNAS(NeurIPS'2019)](/configs/nas/detnas)

  - [x] [SPOS(ECCV'2020)](/configs/nas/spos)
</details>

- <details open><summary>Pruning</summary>

  - [x] [AutoSlim(NeurIPS'2019)](/configs/pruning/autoslim)
</details>

- <details open><summary>Knowledge Distillation</summary>

  - [x] [CWD(ICCV'2021)](/configs/distill/cwd)

  - [x] [WSLD(ICLR'2021)](/configs/distill/wsld)
</details>

## Installation

Please refer to [get_started.md](/docs/en/get_started.md) for installation.

## Getting Started
Please refer to [train.md](/docs/en/train.md) and [test.md](/docs/en/test.md) for the basic usage of MMRazor. There are also tutorials:

- [overview](/docs/en/tutorials/overview.md)
- [learn about configs](/docs/en/tutorials/learn_about_configs.md)
- [customize architectures](/docs/en/tutorials/customize_architectures.md)
- [customize nas algorithms](/docs/en/tutorials/customize_nas_algorithms.md)
- [customize pruning algorithms](/docs/en/tutorials/customize_pruning_algorithms.md)
- [customize kd algorithms](/docs/en/tutorials/customize_kd_algorithms.md)
- [customize mixed algorithms with our algorithm_components](/docs/en/tutorials/customize_mixed_algorithms_with_out_algorithms_components.md)
- [apply existing algorithms to other existing tasks](/docs/en/tutorials/apply_existing_algorithms_to_new_tasks.md)

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2021mmrazor,
    title={OpenMMLab's Model Compression Toolbox and Benchmark},
    author={MMRazor Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmrazor}},
    year={2021}
}
```

## Contributing

We appreciate all contributions to improve MMRazor.
Please refer to [CONTRUBUTING.md](/.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMRazor is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new model compression methods.

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM Installs OpenMMLab Packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab toolbox for text detection, recognition and understanding.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMlab toolkit for generative models.
- [MMFlow](https://github.com/open-mmlab/mmflow) OpenMMLab optical flow toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab FewShot Learning Toolbox and Benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D Human Parametric Model Toolbox and Benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning Toolbox and Benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab Model Compression Toolbox and Benchmark.
