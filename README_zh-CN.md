<div align="center">
  <img src="./resources/mmrazor-logo.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">OpenMMLab 官网</font></b>
    <sup>
      <a href="https://openmmlab.com">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">OpenMMLab 开放平台</font></b>
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

[📘使用文档](https://mmrazor.readthedocs.io/) |
[🛠️安装教程](https://mmrazor.readthedocs.io/en/latest/get_started.html) |
[👀模型库](https://mmrazor.readthedocs.io/en/latest/model_zoo.html) |
[🤔报告问题](https://github.com/open-mmlab/mmrazor/issues/new/choose)

</div>

<!--中/英 文档切换-->

<div align="center">

[English](/README.md) | 简体中文

</div>

## 说明

MMRazor是一个可用于模型瘦身和AutoML的模型压缩工具箱，包含了3种主流的技术：

- 网络结构搜索 (NAS)
- 模型剪枝
- 知识蒸馏 (KD)
- 量化 (下个版本发布)

MMRazor是[OpenMMLab](https://openmmlab.com/)项目的一部分。

主要特性

- **兼容性**

  MMRazor和OpenMMLab有着类似的架构设计，并且实现了轻量化算法和视觉任务间轻耦合，因此很容易应用于OpenMMLab中其他的项目。

- **灵活性**

  多种轻量化算法可以以一种即插即用的方式来组合使用，从而搭建出功能更强大的系统。

- **便利性**

  得益于更好的模块化设计，开发者仅用修改少量代码，甚至只用修改配置文件即可实现新的轻量化算法。

下面是MMRazor设计和实现的概括图, 如果想了解更多的细节，请参考 [tutorials](/docs/en/tutorials/Tutorial_1_overview.md)。

<div align="center">
  <img src="resources/design_and_implement.png" style="zoom:100%"/>
</div>
<br />

## 更新日志

MMRazor v0.3.1 版本已经在 2022.5.4 发布。

## 基准测试和模型库

测试结果可以在 [模型库](docs/en/model_zoo.md) 中找到.

已经支持的算法：

Neural Architecture Search

- [x] [DARTS(ICLR'2019)](configs/nas/darts)

- [x] [DetNAS(NeurIPS'2019)](configs/nas/detnas)

- [x] [SPOS(ECCV'2020)](configs/nas/spos)

Pruning

- [x] [AutoSlim(NeurIPS'2019)](/configs/pruning/autoslim)

Knowledge Distillation

- [x] [CWD(ICCV'2021)](/configs/distill/cwd)

- [x] [WSLD(ICLR'2021)](/configs/distill/wsld)

## 安装

MMRazor 依赖 [PyTorch](https://pytorch.org/) 和 [MMCV](https://github.com/open-mmlab/mmcv)。

请参考[get_started.md](/docs/en/get_started.md)获取更详细的安装指南。

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

## 贡献指南

我们感谢所有的贡献者为改进和提升 MMRazor 所作出的努力。
请参考[贡献指南](/.github/CONTRIBUTING.md)来了解参与项目贡献的相关指引。

## 致谢

MMRazor 是一款由来自不同高校和企业的研发人员共同参与贡献的开源项目。我们感谢所有为项目提供算法复现和新功能支持的贡献者，以及提供宝贵反馈的用户。 我们希望这个工具箱和基准测试可以为社区提供灵活的代码工具，供用户复现已有算法并开发自己的新模型压缩算法，从而不断为开源社区提供贡献。

## 引用

如果您发现此项目对您的研究有用，请考虑引用：

```BibTeX
@misc{2021mmrazor,
    title={OpenMMLab Model Compression Toolbox and Benchmark},
    author={MMRazor Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmrazor}},
    year={2021}
}
```

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## OpenMMLab 的其他项目

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab 计算机视觉基础库
- [MIM](https://github.com/open-mmlab/mim): MIM 是 OpenMMlab 项目、算法、模型的统一入口
- [MMClassification](https://github.com/open-mmlab/mmclassification): OpenMMLab 图像分类工具箱
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab 目标检测工具箱
- [MMDetection3D](https://github.com/open-mmlab/mmdetection3d): OpenMMLab 新一代通用 3D 目标检测平台
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab 旋转框检测工具箱与测试基准
- [MMYOLO](https://github.com/open-mmlab/mmyolo): OpenMMLab YOLO 系列工具箱与测试基准
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation): OpenMMLab 语义分割工具箱
- [MMOCR](https://github.com/open-mmlab/mmocr): OpenMMLab 全流程文字检测识别理解工具箱
- [MMPose](https://github.com/open-mmlab/mmpose): OpenMMLab 姿态估计工具箱
- [MMHuman3D](https://github.com/open-mmlab/mmhuman3d): OpenMMLab 人体参数化模型工具箱与测试基准
- [MMSelfSup](https://github.com/open-mmlab/mmselfsup): OpenMMLab 自监督学习工具箱与测试基准
- [MMRazor](https://github.com/open-mmlab/mmrazor): OpenMMLab 模型压缩工具箱与测试基准
- [MMFewShot](https://github.com/open-mmlab/mmfewshot): OpenMMLab 少样本学习工具箱与测试基准
- [MMAction2](https://github.com/open-mmlab/mmaction2): OpenMMLab 新一代视频理解工具箱
- [MMTracking](https://github.com/open-mmlab/mmtracking): OpenMMLab 一体化视频目标感知平台
- [MMFlow](https://github.com/open-mmlab/mmflow): OpenMMLab 光流估计工具箱与测试基准
- [MMEditing](https://github.com/open-mmlab/mmediting): OpenMMLab 图像视频编辑工具箱
- [MMGeneration](https://github.com/open-mmlab/mmgeneration): OpenMMLab 图片视频生成模型工具箱
- [MMDeploy](https://github.com/open-mmlab/mmdeploy): OpenMMLab 模型部署框架

## 欢迎加入 OpenMMLab 社区

扫描下方的二维码可关注 OpenMMLab 团队的 [知乎官方账号](https://www.zhihu.com/people/openmmlab)，加入 OpenMMLab 团队的 [官方交流 QQ 群](https://jq.qq.com/?_wv=1027&k=aCvMxdr3)，添加OpenMMLab 官方小助手微信，加入 MMSelfSup 微信社区。

<div align="center">
<img src="./resources/zhihu_qrcode.jpg" height="400"/>  <img src="./resources/qq_group_qrcode.jpg" height="400"/> <img src="./resources/xiaozhushou_weixin_qrcode.jpeg" height="300"/>
</div>

我们会在 OpenMMLab 社区为大家

- 📢 分享 AI 框架的前沿核心技术
- 💻 解读 PyTorch 常用模块源码
- 📰 发布 OpenMMLab 的相关新闻
- 🚀 介绍 OpenMMLab 开发的前沿算法
- 🏃 获取更高效的问题答疑和意见反馈
- 🔥 提供与各行各业开发者充分交流的平台

干货满满 📘，等你来撩 💗，OpenMMLab 社区期待您的加入 👬
