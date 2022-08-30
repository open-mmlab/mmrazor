# RKD

> [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068)

<!-- [ALGORITHM] -->

## Abstract

Knowledge distillation aims at transferring knowledge acquired
in one model (a teacher) to another model (a student) that is
typically smaller. Previous approaches can be expressed as
a form of training the student to mimic output activations of
individual data examples represented by the teacher. We introduce
a novel approach, dubbed relational knowledge distillation (RKD),
that transfers mutual relations of data examples instead.
For concrete realizations of RKD, we propose distance-wise and
angle-wise distillation losses that penalize structural differences
in relations. Experiments conducted on different tasks show that the
proposed method improves educated student models with a significant margin.
In particular for metric learning, it allows students to outperform their
teachers' performance, achieving the state of the arts on standard benchmark datasets.

![pipeline](https://user-images.githubusercontent.com/88702197/187424092-b58742aa-6724-4a89-8d28-62960efb58b4.png)

## Results and models

### Classification

| Location | Dataset  |                                                   Teacher                                                    |                                                   Student                                                    |  Acc  | Acc(T) | Acc(S) |                        Config                        | Download                                                                                                                                                                                                                                                                                                                                                                                    |
| :------: | :------: | :----------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------: | :---: | :----: | :----: | :--------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
|   neck   | ImageNet | [resnet34](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet34_8xb32_in1k.py) | [resnet18](https://github.com/open-mmlab/mmclassification/blob/master/configs/resnet/resnet18_8xb32_in1k.py) | 70.23 | 73.62  | 69.90  | [config](./rkd_neck_resnet34_resnet18_8xb32_in1k.py) | [teacher](https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_b16x8_cifar10_20210528-a8aa36a6.pth) \|[model](https://download.openmmlab.com/mmrazor/v0.3/distill/rkd/rkd_neck_resnet34_resnet18_8xb32_in1k_acc-70.23_20220401-f25700ac.pth) \| [log](https://download.openmmlab.com/mmrazor/v0.3/distill/rkd/rkd_neck_resnet34_resnet18_8xb32_in1k_20220312_130419.log.json) |

## Citation

```latex
@inproceedings{park2019relational,
  title={Relational knowledge distillation},
  author={Park, Wonpyo and Kim, Dongju and Lu, Yan and Cho, Minsu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3967--3976},
  year={2019}
}
```
