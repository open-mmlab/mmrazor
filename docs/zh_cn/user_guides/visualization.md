## 可视化

## 特征图可视化

<div align=center>
<img src="https://user-images.githubusercontent.com/41630003/197720299-774b202c-fecc-414b-9f31-499092caee18.jpg" width="1000" alt="image"/>
</div>
可视化可以给深度学习的模型训练和测试过程提供直观解释。

MMRazor 中，将使用 MMEngine 提供的 `Visualizer` 可视化器搭配 MMRazor 自带的 `Recorder`组件的数据记录功能，进行特征图可视化，其具备如下功能：

- 支持基础绘图接口以及特征图可视化。
- 支持选择模型中的任意位点来得到特征图，包含 `pixel_wise_max` ，`squeeze_mean` ， `select_max` ， `topk` 四种显示方式，用户还可以使用 `arrangement` 自定义特征图显示的布局方式。

## 特征图绘制

你可以调用 `tools/visualizations/vis_configs/feature_visualization.py` 来简单快捷地得到单张图片单个模型的可视化结果。

为了方便理解，将其主要参数的功能梳理如下：

- `img`：选择要用于特征图可视化的图片，支持单张图片或者图片路径列表。

- `config`：选择算法的配置文件。

- `vis_config`：可视化功能需借助可配置的 `Recorder` 组件获取模型中用户自定义位点的特征图，
  用户可以将 `Recorder` 相关配置文件放入 `vis_config` 中。 MMRazor提供了对backbone及neck
  输出进行可视化对应的config文件，详见 `configs/visualizations`

- `checkpoint`：选择对应算法的权重文件。

- `--out-file`：将得到的特征图保存到本地，并指定路径和文件名。若没有选定，则会直接显示特征图。

- `--device`：指定用于推理图片的硬件，`--device cuda：0`  表示使用第 1 张 GPU 推理，`--device cpu` 表示用 CPU 推理。

- `--repo`：模型对应的算法库。`--repo mmdet` 表示模型为检测模型。

- `--use-norm`：是否将获取的特征图进行batch normalization后再显示。

- `--overlaid`：是否将特征图覆盖在原图之上。若设为True，考虑到输入的特征图通常非常小，函数默认将特征图进行上采样后方便进行可视化。

- `--channel-reduction`：输入的 Tensor 一般是包括多个通道的，`channel_reduction` 参数可以将多个通道压缩为单通道，然后和图片进行叠加显示，有以下三个参数可以设置：

  - `pixel_wise_max`：将输入的 C 维度采用 max 函数压缩为一个通道，输出维度变为 (1, H, W)。
  - `squeeze_mean`：将输入的 C 维度采用 mean 函数压缩为一个通道，输出维度变成 (1, H, W)。
  - `select_max`：从输入的 C 维度中先在空间维度 sum，维度变成 (C, )，然后选择值最大的通道。
  - `None`：表示不需要压缩，此时可以通过 `topk` 参数可选择激活度最高的 `topk` 个特征图显示。

- `--topk`：只有在 `channel_reduction` 参数为 `None` 的情况下， `topk` 参数才会生效，其会按照激活度排序选择 `topk` 个通道，然后和图片进行叠加显示，并且此时会通过 `--arrangement` 参数指定显示的布局，该参数表示为一个数组，两个数字需要以空格分开，例如： `--topk 5 --arrangement 2 3` 表示以 `2行 3列` 显示激活度排序最高的 5 张特征图， `--topk 7 --arrangement 3 3` 表示以 `3行 3列` 显示激活度排序最高的 7 张特征图。

  - 如果 topk 不是 -1，则会按照激活度排序选择 topk 个通道显示。
  - 如果 topk = -1，此时通道 C 必须是 1 或者 3 表示输入数据是图片，否则报错提示用户应该设置 `channel_reduction` 来压缩通道。

- `--arrangement`：特征图的排布。当 `channel_reduction` 不是None且topk > 0时才会有用。

- `--resize-shape`：当`--overlaid`为True时，是否需要将原图和特征图resize为某一尺寸。

- `--cfg-options`：由于不同算法库的visualizer拥有特例化的add_datasample方法，如mmdet的visualizer
  拥有 `pred_score_thr` 作为输入参数，可以在`--cfg-options`加入一些特例化的设置。

类似的，用户可以通过调用 `tools/visualizations/vis_configs/feature_diff_visualization.py` 来得到
单张图片两个模型的特征差异可视化结果，用法与上述类似，差异为：

- `config1` / `config2`：选择算法1/2的配置文件。
- `checkpoint1` / `checkpoint2`：选择对应算法1/2的权重文件。

## 用法示例

以预训练好的 RetinaNet-r101 与 RetinaNet-r50 模型为例:

请提前下载 RetinaNet-r101 与 RetinaNet-r50 模型权重到本仓库根路径下：

```shell
cd mmrazor
wget https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r101_fpn_2x_coco/retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth
wget https://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth
```

(1) 将多通道特征图采用 `pixel_wise_max` 参数压缩为单通道并显示, 通过提取 `neck` 层输出进行特征图可视化（这里只显示了前4个stage的特征图）：

```shell
python tools/visualizations/feature_visualization.py \
       tools/visualizations/demo.jpg \
       PATH/TO/THE/CONFIG \
       tools/visualizations/vis_configs/fpn_feature_visualization.py \
       retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth \
       --repo mmdet --use-norm --overlaid
       --channel-reduction pixel_wise_max
```

<div align=center>
<img src="https://user-images.githubusercontent.com/41630003/197720372-08e29a02-21ce-46a4-910a-97aabe7ec796.jpg" width="800" alt="image"/>
</div>

(2) 将多通道特征图采用 `select_max` 参数压缩为单通道并显示, 通过提取 `neck` 层输出进行特征图可视化（这里只显示了前4个stage的特征图）：

```shell
python tools/visualizations/feature_visualization.py \
       tools/visualizations/demo.jpg \
       PATH/TO/THE/CONFIG \
       tools/visualizations/vis_configs/fpn_feature_visualization.py \
       retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth \
       --repo mmdet --overlaid
       --channel-reduction select_max
```

<div align=center>
<img src="https://user-images.githubusercontent.com/41630003/197720581-0ed2fd5a-e07d-4320-90e7-adbe0f05fd41.jpg" width="800" alt="image"/>
</div>

(3) 将多通道特征图采用 `squeeze_mean` 参数压缩为单通道并显示, 通过提取 `neck` 层输出进行特征图可视化（这里只显示了前4个stage的特征图）：

```shell
python tools/visualizations/feature_visualization.py \
       tools/visualizations/demo.jpg \
       PATH/TO/THE/CONFIG \
       tools/visualizations/vis_configs/fpn_feature_visualization.py \
       retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth \
       --repo mmdet --overlaid
       --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/41630003/197720659-b25fcbcf-c5c5-45a6-965d-5336d136acde.jpg" width="800" alt="image"/>
</div>

(4) 将多通道特征图采用 `squeeze_mean` 参数压缩为单通道并显示, 通过提取 `neck` 层输出进行特征图可视化（这里只显示了前4个stage的特征图）：

```shell
python tools/visualizations/feature_visualization.py \
       tools/visualizations/demo.jpg \
       PATH/TO/THE/CONFIG \
       tools/visualizations/vis_configs/fpn_feature_visualization.py \
       retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth \
       --repo mmdet --overlaid
       --channel-reduction squeeze_mean
```

<div align=center>
<img src="https://user-images.githubusercontent.com/41630003/197720735-f30299be-2c42-444b-bec6-759723ad43fa.jpg" width="800" alt="image"/>
</div>

(5) 将多通道的两个模型的特征图差异采用 `pixel_wise_max` 参数压缩为单通道并显示, 这里只显示了前4个stage的特征图差异：

```shell
python tools/visualizations/feature_diff_visualization.py \
       tools/visualizations/demo.jpg \
       PATH/TO/THE/CONFIG1 \
       PATH/TO/THE/CONFIG2 \
       tools/visualizations/vis_configs/fpn_feature_diff_visualization.py.py \
       retinanet_r101_fpn_2x_coco_20200131-5560aee8.pth \
       retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth \
       --repo mmdet --use-norm --overlaid
       --channel-reduction pixel_wise_max
```

<div align=center>
<img src="https://user-images.githubusercontent.com/41630003/197720804-be0f3a27-e4d7-4160-b518-33d527114f9f.jpg" width="800" alt="image"/>
</div>
