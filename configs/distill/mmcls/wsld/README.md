# WSLD

> [Rethinking Soft Labels for Knowledge Distillation: A Bias-Variance Tradeoff Perspective](https://arxiv.org/abs/2102.00650)

<!-- [ALGORITHM] -->

## Abstract

Knowledge distillation is an effective approach to leverage a well-trained network
or an ensemble of them, named as the teacher, to guide the training of a student
network. The outputs from the teacher network are used as soft labels for supervising the training of a new network. Recent studies (Muller et al., 2019; Yuan ¨
et al., 2020) revealed an intriguing property of the soft labels that making labels
soft serves as a good regularization to the student network. From the perspective of statistical learning, regularization aims to reduce the variance, however
how bias and variance change is not clear for training with soft labels. In this
paper, we investigate the bias-variance tradeoff brought by distillation with soft
labels. Specifically, we observe that during training the bias-variance tradeoff
varies sample-wisely. Further, under the same distillation temperature setting, we
observe that the distillation performance is negatively associated with the number of some specific samples, which are named as regularization samples since
these samples lead to bias increasing and variance decreasing. Nevertheless, we
empirically find that completely filtering out regularization samples also deteriorates distillation performance. Our discoveries inspired us to propose the novel
weighted soft labels to help the network adaptively handle the sample-wise biasvariance tradeoff. Experiments on standard evaluation benchmarks validate the
effectiveness of our method.

<img width="1032" alt="pipeline" src="https://user-images.githubusercontent.com/88702197/187424195-a3ea3d72-5ee7-4ffc-b562-65677076c18e.png">

## Results and models

### Classification

| Location | Dataset  |                                                   Teacher                                                    |                                                   Student                                                    |  Acc  | Acc(T) | Acc(S) |                          Config                           | Download                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| :------: | :------: | :----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| cls head | ImageNet | [resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb32_in1k.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | 71.54 | 73.62  | 69.90  | [config](./wsld_cls_head_resnet34_resnet18_8xb32_in1k.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \|[model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/distill/wsld/wsld_cls_head_resnet34_resnet18_8xb32_in1k/wsld_cls_head_resnet34_resnet18_8xb32_in1k_acc-71.54_20211222-91f28cf6.pth?versionId=CAEQHxiBgMC6memK7xciIGMzMDFlYTA4YzhlYTRiMTNiZWU0YTVhY2I5NjVkMjY2) \| [log](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v0.1/distill/wsld/wsld_cls_head_resnet34_resnet18_8xb32_in1k/wsld_cls_head_resnet34_resnet18_8xb32_in1k_20211221_181516.log.json?versionId=CAEQHxiBgIDLmemK7xciIGNkM2FiN2Y4N2E5YjRhNDE4NDVlNmExNDczZDIxN2E5) |

## Citation

```latex
@inproceedings{zhou2021wsl,
  title={Rethinking soft labels for knowledge distillation: a bias-variance tradeoff perspective},
  author={Helong, Zhou and Liangchen, Song and Jiajie, Chen and Ye, Zhou and Guoli, Wang and Junsong, Yuan and Qian Zhang},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
