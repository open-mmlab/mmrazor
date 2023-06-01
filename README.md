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

<!--算法库 Badges-->

[![PyPI](https://img.shields.io/pypi/v/mmrazor)](https://pypi.org/project/mmrazor)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmrazor.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmrazor/workflows/build/badge.svg)](https://github.com/open-mmlab/mmrazor/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmrazor/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmrazor)
[![license](https://img.shields.io/github/license/open-mmlab/mmrazor.svg)](https://github.com/open-mmlab/mmrazor/blob/master/LICENSE)
[![open issues](https://isitmaintained.com/badge/open/open-mmlab/mmrazor.svg)](https://github.com/open-mmlab/mmrazor/issues)
[![issue resolution](https://isitmaintained.com/badge/resolution/open-mmlab/mmrazor.svg)](https://github.com/open-mmlab/mmrazor/issues)

<!--快速链接-->

<!--Note:请根据各算法库自身情况设置项目和链接-->

[📘Documentation](https://mmrazor.readthedocs.io/en/latest/) |
[🛠️Installation](https://mmrazor.readthedocs.io/en/latest/get_started/installation.html) |
[👀Model Zoo](https://mmrazor.readthedocs.io/en/latest/get_started/model_zoo.html) |
[🤔Reporting Issues](https://github.com/open-mmlab/mmrazor/issues/new/choose)

</div>

<!--中/英 文档切换-->

<div align="center">

English | [简体中文](README_zh-CN.md)

<div align="center">
  <a href="https://openmmlab.medium.com/" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218352562-cdded397-b0f3-4ca1-b8dd-a60df8dca75b.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://discord.gg/raweFPmdzG" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218347213-c080267f-cbb6-443e-8532-8e1ed9a58ea9.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://twitter.com/OpenMMLab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346637-d30c8a0f-3eba-4699-8131-512fb06d46db.png" width="3%" alt="" /></a>
  <img src="https://user-images.githubusercontent.com/25839884/218346358-56cc8e2f-a2b8-487f-9088-32480cceabcf.png" width="3%" alt="" />
  <a href="https://www.youtube.com/openmmlab" style="text-decoration:none;">
    <img src="https://user-images.githubusercontent.com/25839884/218346691-ceb2116a-465a-40af-8424-9f30d2348ca9.png" width="3%" alt="" /></a>
</div>

</div>

**:star: MMRazor for Large Models** is Available Now! Please refer to [MMRazorLarge](projects/mmrazor_large/README.md)

## Introduction

MMRazor is a model compression toolkit for model slimming and AutoML, which includes 4 mainstream technologies:

- Neural Architecture Search (NAS)
- Pruning
- Knowledge Distillation (KD)
- Quantization

It is a part of the [OpenMMLab](https://openmmlab.com/) project.

Major features:

- **Compatibility**

  MMRazor can be easily applied to various projects in OpenMMLab, due to the similar architecture design of OpenMMLab as well as the decoupling of slimming algorithms and vision tasks.

- **Flexibility**

  Different algorithms, e.g., NAS, pruning and KD, can be incorporated in a plug-n-play manner to build a more powerful system.

- **Convenience**

  With better modular design, developers can implement new model compression algorithms with only a few codes, or even by simply modifying config files.

About MMRazor's design and implementation, please refer to [tutorials](https://mmrazor.readthedocs.io/en/latest/get_started/overview.html) for more details.

## Latest Updates

**The default branch is now `main` and the code on the branch has been upgraded to v1.0.0. The old `master` branch code now exists on the 0.x branch**

MMRazor v1.0.0 was released in 2023-4-24, Major updates from 1.0.0rc2 include:

1. MMRazor quantization is released.
2. Add a new pruning algorithm named GroupFisher.
3. Support distilling rtmdet with MMRazor.

To know more about the updates in MMRazor 1.0, please refer to [Changelog](https://mmrazor.readthedocs.io/en/latest/notes/changelog.html) for more details!

## Benchmark and model zoo

Results and models are available in the [model zoo](https://mmrazor.readthedocs.io/en/latest/get_started/model_zoo.html).

Supported algorithms:

<details open>
<summary>Neural Architecture Search</summary>

- [x] [DARTS(ICLR'2019)](configs/nas/mmcls/darts)

- [x] [DetNAS(NeurIPS'2019)](configs/nas/mmdet/detnas)

- [x] [SPOS(ECCV'2020)](configs/nas/mmcls/spos)

</details>

<details open>
<summary>Pruning</summary>

- [x] [AutoSlim(NeurIPS'2019)](/configs/pruning/mmcls/autoslim)

- [x] [L1-norm](/configs/pruning/mmcls/l1-norm)

- [x] [Group Fisher](/configs/pruning/base/group_fisher)

- [x] [DMCP](/configs/pruning/mmcls/dmcp)

</details>

<details open>
<summary>Knowledge Distillation</summary>

- [x] [CWD(ICCV'2021)](/configs/distill/mmdet/cwd)

- [x] [WSLD(ICLR'2021)](/configs/distill/mmcls/wsld)

- [x] [ABLoss](/configs/distill/mmcls/abloss)

- [x] [BYOT](/configs/distill/mmcls/byot)

- [x] [DAFL](/configs/distill/mmcls/dafl)

- [x] [DFAD](/configs/distill/mmcls/dfad)

- [x] [DKD](/configs/distill/mmcls/dkd)

- [x] [Factor Transfer](/configs/distill/mmcls/factor_transfer)

- [x] [FitNets](/configs/distill/mmcls/fitnets)

- [x] [KD](/configs/distill/mmcls/kd)

- [x] [OFD](/configs/distill/mmcls/ofd)

- [x] [RKD](/configs/distill/mmcls/rkd)

- [x] [ZSKT](/configs/distill/mmcls/zskt)

- [x] [FBKD](/configs/distill/mmdet/fbkd)

</details>

<details open>
<summary>Quantization</summary>

- [x] [PTQ](/configs/quantization/ptq/base)

- [x] [QAT](/configs/quantization/qat/base)

- [x] [LSQ](/configs/quantization/qat/lsq)

</details>

## Installation

MMRazor depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMEngine](https://github.com/open-mmlab/mmengine).

Please refer to [installation.md](https://mmrazor.readthedocs.io/en/latest/get_started/installation.html) for more detailed instruction.

## Getting Started

Please refer to [user guides](https://mmrazor.readthedocs.io/en/latest/user_guides/index.html) for the basic usage of MMRazor. There are also [advanced guides](https://mmrazor.readthedocs.io/en/latest/advanced_guides/index.html):

## Contributing

We appreciate all contributions to improve MMRazor.
Please refer to [CONTRUBUTING.md](https://mmrazor.readthedocs.io/en/latest/notes/contribution_guide.html) for the contributing guideline.

## Acknowledgement

MMRazor is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new model compression methods.

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

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Projects in OpenMMLab

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MIM](https://github.com/open-mmlab/mim): MIM installs OpenMMLab packages.
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab image classification toolbox and benchmark.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab's next-generation platform for general 3D object detection.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO series toolbox and benchmark.
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
