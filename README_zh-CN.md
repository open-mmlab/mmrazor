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

文档: https://mmrazor.readthedocs.io/

[English](/README.md) | 简体中文

## 说明

MMRazor 是一个基于 PyTorch 的模型压缩开源工具箱. 它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分.


<details open><summary>主要特性</summary>

- **多合一**

  MMrazor 实现了4中主流的技术：1）网络结构搜索；2）模型剪枝；3）知识蒸馏；4）量化（很快就会支持），并支持不同算法间的
  互相组合。

- **通用计算机视觉模型压缩工具包**

  得益于 OpenMMLab 开源生态，MMRazor 中的算法可快速应用到不同的任务上，使模型压缩算法的开发变的一劳永逸。

- **压缩算法和任务模型的解耦**

  MMrazor 具有多种内置的自动化机制，允许开发者在不修改原始模型代码的情况下实现模型压缩算法，比如：
  - 可以 code-free 地获取模型中间层输出结果；
  - 可以 code-free 地替换模型中的某些 OP；
  - 自动获得并分析 nn.Module 间连接
  - ......

- **灵活和模块化设计**

  MMRazor 将模型压缩算法分解为不同的组件，使得构建新算法变得更加容易和灵活。

</details>

下面是MMRazor设计和实现的概括图, 如果想了解更多的细节，请参考 [tutorials](/docs/en/tutorials/Tutorial_1_overview.md)。
<div align="center">
  <img src="resources/design_and_implement.png" style="zoom:100%"/>
</div>
<br />

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 更新日志

v0.1.0 版本已经在 2021.12.23 发布

## 基准测试和模型库

测试结果可以在 [模型库](/docs/en/model_zoo.md) 中找到.

已经支持的算法:

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

## 安装

请参考 [get_started.md](/docs/en/get_started.md) 进行安装。

## 快速入门
请参考 [get_started.md](/docs/en/get_started.md) 学习 MMRazor 的基本使用。 我们也提供了一些进阶教程:

- [overview](/docs/en/tutorials/Tutorial_1_overview.md)
- [learn about configs](/docs/en/tutorials/Tutorial_2_learn_about_configs.md)
- [customize architectures](/docs/en/tutorials/Tutorial_3_customize_architectures.md)
- [customize nas algorithms](/docs/en/tutorials/Tutorial_4_customize_nas_algorithms.md)
- [customize pruning algorithms](/docs/en/tutorials/Tutorial_5_customize_pruning_algorithms.md)
- [customize kd algorithms](/docs/en/tutorials/Tutorial_6_customize_kd_algorithms.md)
- [customize mixed algorithms with our algorithm_components](/docs/en/tutorials/Tutorial_7_customize_mixed_algorithms_with_out_algorithms_components.md)
- [apply existing algorithms to other existing tasks](/docs/en/tutorials/Tutorial_8_apply_existing_algorithms_to_new_tasks.md)

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 MMRazor。

```BibTeX
@misc{2021mmrazor,
    title={OpenMMLab Model Compression Toolbox and Benchmark},
    author={MMRazor Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmrazor}},
    year={2021}
}
```

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMRazor 所作出的努力.
请参考[贡献指南](/.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMRazor 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型压缩算法，从而不断为开源社区提供贡献.

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具包
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)

<div align="center">
<img src="resources/zhihu_qrcode.jpg" height="400" />  <img src="resources/qq_group_qrcode.jpg" height="400" />
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
