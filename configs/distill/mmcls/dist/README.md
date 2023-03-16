# KD

> [Knowledge Distillation from A Stronger Teacher](https://arxiv.org/abs/2205.10536)

<!-- [ALGORITHM] -->

## Abstract

Unlike existing knowledge distillation methods focus on the baseline settings, where the teacher models and training strategies are not that strong and competing as state-of-the-art approaches, this paper presents a method dubbed DIST to distill better from a stronger teacher. We empirically find that the discrepancy of predictions between the student and a stronger teacher may tend to be fairly severer. As a result, the exact match of predictions in KL divergence would disturb the training and make existing methods perform poorly. In this paper, we show that simply preserving the relations between the predictions of teacher and student would suffice, and propose a correlation-based loss to capture the intrinsic inter-class relations from the teacher explicitly. Besides, considering that different instances have different semantic similarities to each class, we also extend this relational match to the intra-class level. Our method is simple yet practical, and extensive experiments demonstrate that it adapts well to various architectures, model sizes and training strategies, and can achieve state-of-the-art performance consistently on image classification, object detection, and semantic segmentation tasks. Code is available at: this https URL .

## Results and models

### Classification

| Location | Dataset  |      Teacher      |      Student      |  Acc  | Acc(T) | Acc(S) |       Config        | Download                                                         |
| :------: | :------: | :---------------: | :---------------: | :---: | :----: | :----: | :-----------------: | :--------------------------------------------------------------- |
|  logits  | ImageNet | [resnet34][r34_c] | [resnet18][r18_c] | 71.61 | 73.62  | 69.90  | [config][distill_c] | [teacher][r34_pth] \| [model][distill_pth] \| [log][distill_log] |

**Note**

There are fluctuations in the results of the experiments of DIST loss. For example, we run three times of the official code of DIST and get three different results.

| Time | Top-1 |
| ---- | ----- |
| 1th  | 71.69 |
| 2nd  | 71.82 |
| 3rd  | 71.90 |

## Citation

```latex
@article{huang2022knowledge,
  title={Knowledge Distillation from A Stronger Teacher},
  author={Huang, Tao and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang},
  journal={arXiv preprint arXiv:2205.10536},
  year={2022}
}
```

[distill_c]: ./dist_logits_resnet34_resnet18_8xb32_in1k.py
[distill_log]: https://download.openmmlab.com/mmrazor/v1/distillation/dist_logits_resnet34_resnet18_8xb32_in1k.json
[distill_pth]: https://download.openmmlab.com/mmrazor/v1/distillation/dist_logits_resnet34_resnet18_8xb32_in1k.pth
[r18_c]: https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/resnet/resnet18_8xb32_in1k.py
[r34_c]: https://github.com/open-mmlab/mmclassification/blob/dev-1.x/configs/resnet/resnet34_8xb32_in1k.py
[r34_pth]: https://download.openmmlab.com/mmclassification/v0/resnet/resnet34_8xb32_in1k_20210831-f257d4e6.pth
