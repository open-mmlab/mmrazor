# FitNets

> [FitNets: Hints for Thin Deep Nets](https://arxiv.org/abs/1412.6550)

<!-- [ALGORITHM] -->

## Abstract

While depth tends to improve network performances, it also makes gradient-based
training more difficult since deeper networks tend to be more non-linear. The recently
proposed knowledge distillation approach is aimed at obtaining small and fast-to-execute
models, and it has shown that a student network could imitate the soft output of a larger
teacher network or ensemble of networks. In this paper, we extend this idea to allow the
training of a student that is deeper and thinner than the teacher, using not only the outputs
but also the intermediate representations learned by the teacher as hints to improve the
training process and final performance of the student. Because the student intermediate hidden
layer will generally be smaller than the teacher's intermediate hidden layer, additional parameters
are introduced to map the student hidden layer to the prediction of the teacher hidden layer. This
allows one to train deeper students that can generalize better or run faster, a trade-off that is
controlled by the chosen student capacity. For example, on CIFAR-10, a deep student network with
almost 10.4 times less parameters outperforms a larger, state-of-the-art teacher network.

<img width="743" alt="pipeline" src="https://user-images.githubusercontent.com/88702197/187423686-68719140-a978-4a19-a684-42b1d793d1fb.png">

## Results and models

### Classification

|     Location      | Dataset  |                                                   Teacher                                                    |                                                   Student                                                    |  Acc  | Acc(T) | Acc(S) |                               Config                                | Download                                                                                                                                                                                                                                                                                                                                                                                                     |
| :---------------: | :------: | :----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :-----------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| backbone & logits | ImageNet | [resnet50](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet50_8xb32_in1k.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | 70.58 | 76.55  | 69.90  | [config](./fitnets_backbone_logits_resnet50_resnet18_8xb32_in1k.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/FieNets/fitnets_backbone_logits_resnet50_resnet18_8xb32_in1k_20220830_155608-00ccdbe2.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/FieNets/fitnets_backbone_logits_resnet50_resnet18_8xb32_in1k_20220830_155608-00ccdbe2.json) |

## Citation

```latex
@inproceedings{DBLP:journals/corr/RomeroBKCGB14,
  author    = {Adriana Romero, Nicolas Ballas, Samira Ebrahimi Kahou, Antoine Chassang, Carlo Gatta and Yoshua Bengio},
  editor    = {Yoshua Bengio and Yann LeCun},
  title     = {FitNets: Hints for Thin Deep Nets},
  booktitle = {3rd International Conference on Learning Representations, {ICLR} 2015,
               San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings},
  year      = {2015},
  url       = {http://arxiv.org/abs/1412.6550},
  timestamp = {Thu, 25 Jul 2019 14:25:38 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/RomeroBKCGB14.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
