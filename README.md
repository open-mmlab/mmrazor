<div align="center">
  <img src="resources/mmrazor-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab website</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab platform</font></b>
    <sup>
      <a href="https://platform.openmmlab.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>
  <div>&nbsp;</div>
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

MMRazor is a model compression toolkit for model slimming and AutoML, which includes 3 mainstream technologies:

- Neural Architecture Search (NAS)
- Pruning
- Knowledge Distillation (KD)
- Quantization (in the next release)

It is a part of the [OpenMMLab](https://openmmlab.com/) project.

Major features:
- **Compatibility**

  MMRazor can be easily applied to various projects in OpenMMLab, due to the similar architecture design of OpenMMLab as well as the decoupling of slimming algorithms and vision tasks.

- **Flexibility**

  Different algorithms, e.g., NAS, pruning and KD, can be incorporated in a plug-n-play manner to build a more powerful system.

- **Convenience**

  With better modular design, developers can implement new model compression algorithms with only a few codes, or even by simply modifying config files.

Below is an overview of MMRazor's design and implementation, please refer to [tutorials](/docs/en/tutorials/Tutorial_1_overview.md) for more details.
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

## Installation

MMRazor depends on [PyTorch](https://pytorch.org/) and [MMCV](https://github.com/open-mmlab/mmcv).
Below are quick steps for installation.
Please refer to [get_started.md](/docs/en/get_started.md) for more detailed instruction and [dataset_prepare.md](docs/en/dataset_prepare.md) for dataset preparation.

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmrazor.git
cd mmrazor
pip install -v -e .  # or "python setup.py develop"
```

## Getting Started
Please refer to [train.md](/docs/en/train.md) and [test.md](/docs/en/test.md) for the basic usage of MMRazor. There are also tutorials:

- [overview](/docs/en/tutorials/Tutorial_1_overview.md)
- [learn about configs](/docs/en/tutorials/Tutorial_2_learn_about_configs.md)
- [customize architectures](/docs/en/tutorials/Tutorial_3_customize_architectures.md)
- [customize nas algorithms](/docs/en/tutorials/Tutorial_4_customize_nas_algorithms.md)
- [customize pruning algorithms](/docs/en/tutorials/Tutorial_5_customize_pruning_algorithms.md)
- [customize kd algorithms](/docs/en/tutorials/Tutorial_6_customize_kd_algorithms.md)
- [customize mixed algorithms with our algorithm_components](/docs/en/tutorials/Tutorial_7_customize_mixed_algorithms_with_out_algorithms_components.md)
- [apply existing algorithms to other existing tasks](/docs/en/tutorials/Tutorial_8_apply_existing_algorithms_to_new_tasks.md)

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2021mmrazor,
    title={OpenMMLab Model Compression Toolbox and Benchmark},
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
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab semantic segmentation toolbox and benchmark.
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab text detection, recognition, and understanding toolbox.
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab pose estimation toolbox and benchmark.
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 3D human parametric model toolbox and benchmark.
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab self-supervised learning toolbox and benchmark.
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab model compression toolbox and benchmark.
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab fewshot learning toolbox and benchmark.
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab's next-generation action understanding toolbox and benchmark.
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab video perception toolbox and benchmark.
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab optical flow toolbox and benchmark.
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab image and video editing toolbox.
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab image and video generative models toolbox.
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab model deployment framework.
