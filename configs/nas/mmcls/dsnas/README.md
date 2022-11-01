# DSNAS

> [DSNAS: Direct Neural Architecture Search without Parameter Retraining](https://arxiv.org/abs/2002.09128.pdf)

<!-- [ALGORITHM] -->

## Abstract

Most existing NAS methods require two-stage parameter optimization.
However, performance of the same architecture in the two stages correlates poorly.
Based on this observation, DSNAS proposes a task-specific end-to-end differentiable NAS framework that simultaneously optimizes architecture and parameters with a low-biased Monte Carlo estimate. Child networks derived from DSNAS can be deployed directly without parameter retraining.

![pipeline](/docs/en/imgs/model_zoo/dsnas/pipeline.jpg)

## Results and models

### Supernet

| Dataset  | Params(M) | FLOPs (G) | Top-1 Acc (%) | Top-5 Acc (%) |                  Config                   |                                                                                                                         Download                                                                                                                         |     Remarks      |
| :------: | :-------: | :-------: | :-----------: | :-----------: | :---------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------: |
| ImageNet |   3.33    |   0.299   |     73.56     |     91.24     | [config](./dsnas_supernet_8xb128_in1k.py) | [model](https://openmmlab-share.oss-cn-hangzhou.aliyuncs.com/mmrazor/v1/dsnas/dsnas_supernet_8xb128_in1k_20220926_171954-29b87e3a.pth) \| [log](https://download.openmmlab.com/mmrazor/v1/dsnas/dsnas_supernet_8xb128_in1k_20220926_171954-29b87e3a.log) | MMRazor searched |

**Note**:

1. There **might be(not all the case)** some small differences in our experiment in order to be consistent with other repos in OpenMMLab. For example,
   normalize images in data preprocessing; resize by cv2 rather than PIL in training; dropout is not used in network. **Please refer to corresponding config for details.**
2. We convert the official searched checkpoint DSNASsearch240.pth into mmrazor-style and evaluate with pytorch1.8_cuda11.0, Top-1 is 74.1 and Top-5 is 91.51.
3. The implementation of ShuffleNetV2 in official DSNAS is different from OpenMMLab's and we follow the structure design in OpenMMLab. Note that with the
   origin ShuffleNetV2 design in official DSNAS, the Top-1 is 73.92 and Top-5 is 91.59.
4. The finetune stage in our implementation refers to the 'search-from-search' stage mentioned in official DSNAS.
5. We obtain params and FLOPs using `mmrazor.ResourceEstimator`, which may be different from the origin repo.

## Citation

```latex
@inproceedings{hu2020dsnas,
  title={Dsnas: Direct neural architecture search without parameter retraining},
  author={Hu, Shoukang and Xie, Sirui and Zheng, Hehui and Liu, Chunxiao and Shi, Jianping and Liu, Xunying and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12084--12092},
  year={2020}
}
```
