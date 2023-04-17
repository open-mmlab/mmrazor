# DeiT

> [](https://arxiv.org/abs/2012.12877)
> Training data-efficient image transformers & distillation through attention

<!-- [ALGORITHM] -->

## Abstract

Recently, neural networks purely based on attention were shown to address image understanding tasks such as image classification. However, these visual transformers are pre-trained with hundreds of millions of images using an expensive infrastructure, thereby limiting their adoption.   In this work, we produce a competitive convolution-free transformer by training on Imagenet only. We train them on a single computer in less than 3 days. Our reference vision transformer (86M parameters) achieves top-1 accuracy of 83.1% (single-crop evaluation) on ImageNet with no external data.   More importantly, we introduce a teacher-student strategy specific to transformers. It relies on a distillation token ensuring that the student learns from the teacher through attention. We show the interest of this token-based distillation, especially when using a convnet as a teacher. This leads us to report results competitive with convnets for both Imagenet (where we obtain up to 85.2% accuracy) and when transferring to other tasks. We share our code and models.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/143225703-c287c29e-82c9-4c85-a366-dfae30d198cd.png" width="40%"/>
</div>

## Results and models

### Classification

| Dataset  | Model     | Teacher     | Top-1 (%) | Top-5 (%) | Configs                                          | Download                                                                                                                                                                                                                                                                |
| -------- | --------- | ----------- | --------- | --------- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ImageNet | Deit-base | RegNety-160 | 83.24     | 96.33     | [config](deit-base_regnety160_pt-16xb64_in1k.py) | [model](https://download.openmmlab.com/mmrazor/v1/deit/deit-base/deit-base_regnety160_pt-16xb64_in1k_20221011_113403-a67bf475.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/deit/deit-base/deit-base_regnety160_pt-16xb64_in1k_20221011_113403-a67bf475.json) |

```{warning}
Before training, please first install `timm`.

pip install timm
or
git clone https://github.com/rwightman/pytorch-image-models
cd pytorch-image-models && pip install -e .
```

## Citation

```
@InProceedings{pmlr-v139-touvron21a,
  title =     {Training data-efficient image transformers &amp; distillation through attention},
  author =    {Touvron, Hugo and Cord, Matthieu and Douze, Matthijs and Massa, Francisco and Sablayrolles, Alexandre and Jegou, Herve},
  booktitle = {International Conference on Machine Learning},
  pages =     {10347--10357},
  year =      {2021},
  volume =    {139},
  month =     {July}
}
```
