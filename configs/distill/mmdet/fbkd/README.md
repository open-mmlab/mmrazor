# IMPROVE OBJECT DETECTION WITH FEATURE-BASED KNOWLEDGE DISTILLATION: TOWARDS ACCURATE AND EFFICIENT DETECTORS (FBKD)

> [IMPROVE OBJECT DETECTION WITH FEATURE-BASED KNOWLEDGE DISTILLATION: TOWARDS ACCURATE AND EFFICIENT DETECTORS](https://openreview.net/pdf?id=uKhGRvM8QNH)

<!-- [ALGORITHM] -->

## Abstract

Knowledge distillation, in which a student model is trained to mimic a teacher model, has been proved as an effective technique for model compression and model accuracy boosting. However, most knowledge distillation methods, designed for image classification, have failed on more challenging tasks, such as object detection. In this paper, we suggest that the failure of knowledge distillation on object detection is mainly caused by two reasons: (1) the imbalance between pixels of foreground and background and (2) lack of distillation on the relation between different pixels. Observing the above reasons, we propose attention-guided distillation and non-local distillation to address the two problems, respectively. Attention-guided distillation is proposed to find the crucial pixels of foreground objects with attention mechanism and then make the students take more effort to learn their features. Non-local distillation is proposed to enable students to learn not only the feature of an individual pixel but also the relation between different pixels captured by non-local modules. Experiments show that our methods achieve excellent AP improvements on both one-stage and two-stage, both anchor-based and anchor-free detectors. For example, Faster RCNN (ResNet101 backbone) with our distillation achieves 43.9 AP on COCO2017, which is 4.1 higher than the baseline.

<img width="836" alt="pipeline" src="https://user-images.githubusercontent.com/88702197/187424617-6259a7fc-b610-40ae-92eb-f21450dcbaa1.png">

## Results and models

### Detection

| Location | Dataset |                                                              Teacher                                                               |                                                             Student                                                              | box AP | box AP(T) | box AP(S) |                             Config                             | Download                                                                                                                                                                                                                                                                                                                                                                                                                             |
| :------: | :-----: | :--------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------: | :----: | :-------: | :-------: | :------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   neck   |  COCO   | [faster-rcnn_resnet101](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py) | [faster-rcnn_resnet50](https://github.com/open-mmlab/mmdetection/blob/master/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py) |  39.3  |   39.4    |   37.4    | [config](./fbkd_fpn_frcnn_resnet101_frcnn_resnet50_1x_coco.py) | [teacher](https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r101_fpn_1x_coco/faster_rcnn_r101_fpn_1x_coco_20200130-f513f705.pth) \|[model](https://download.openmmlab.com/mmrazor/v1/FBKD/fbkd_fpn_frcnn_resnet101_frcnn_resnet50_1x_coco_20220830_121522-8d7e11df.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/FBKD/fbkd_fpn_frcnn_resnet101_frcnn_resnet50_1x_coco_20220830_121522-8d7e11df.json) |

## Citation

```latex
@inproceedings{DBLP:conf/iclr/ZhangM21,
  author    = {Linfeng Zhang and Kaisheng Ma},
  title     = {Improve Object Detection with Feature-based Knowledge Distillation:
               Towards Accurate and Efficient Detectors},
  booktitle = {9th International Conference on Learning Representations, {ICLR} 2021,
               Virtual Event, Austria, May 3-7, 2021},
  publisher = {OpenReview.net},
  year      = {2021},
  url       = {https://openreview.net/forum?id=uKhGRvM8QNH},
  timestamp = {Wed, 23 Jun 2021 17:36:39 +0200},
  biburl    = {https://dblp.org/rec/conf/iclr/ZhangM21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
