# Train different types algorithms

**Before running our algorithms, you may need to prepare the datasets according to the instructions in the corresponding document.**

**Note**:

- With the help of mmengine, mmrazor unified entered interfaces for various tasks, thus our algorithms will adapt all OpenMMLab upstream repos in theory.

- We dynamically pass arguments `cfg-options` (e.g., `mutable_cfg` in nas algorithm or `channel_cfg` in pruning algorithm) to **avoid the need for a config for each subnet or checkpoint**. If you want to specify different subnets for retraining or testing, you just need to change this argument.

### NAS

Here we take SPOS(Single Path One Shot) as an example. There are three steps to start neural network search(NAS), including **supernet pre-training**, **search for subnet on the trained supernet** and **subnet retraining**.

#### Supernet Pre-training

```Python
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

The usage of optional arguments are the same as corresponding tasks like mmclassification, mmdetection and mmsegmentation.

For example,

```Python
python ./tools/train.py \
  configs/nas/mmcls/spos/spos_shufflenet_supernet_8xb128_in1k.py
  --work-dir $WORK_DIR
```

#### Search for Subnet on The Trained Supernet

```Python
python tools/train.py ${CONFIG_FILE} --cfg-options load_from=${CHECKPOINT_PATH} [optional arguments]
```

For example,

```Python
python ./tools/train.py \
  configs/nas/mmcls/spos/spos_shufflenet_search_8xb128_in1k.py \
  --cfg-options load_from=$STEP1_CKPT \
  --work-dir $WORK_DIR
```

#### Subnet Retraining

```Python
python tools/train.py ${CONFIG_FILE} \
    --cfg-options algorithm.fix_subnet=${MUTABLE_CFG_PATH} [optional arguments]
```

- `MUTABLE_CFG_PATH`: Path of `fix_subnet`. `fix_subnet` represents **config for mutable of the subnet searched out**, used to specify different subnets for retraining. An example for `fix_subnet` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml), and the usage can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/README.md#subnet-retraining-on-imagenet).

For example,

```Python
python ./tools/train.py \
  configs/nas/mmcls/spos/spos_shufflenet_subnet_8xb128_in1k.py \
  --work-dir $WORK_DIR \
  --cfg-options algorithm.fix_subnet=$YAML_FILE_BY_STEP2
```

We note that instead of using `--cfg-options`, you can also directly modify ``` configs/nas/mmcls/spos/``spos_shufflenet_subnet_8xb128_in1k``.py ``` like this:

```Python
fix_subnet = 'configs/nas/mmcls/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml'
model = dict(fix_subnet=fix_subnet)
```

### Pruning

Pruning has three steps, including **supernet pre-training**, **search for subnet on the trained supernet** and **subnet retraining**. The commands of the first two steps are similar to NAS, except that we need to use `CONFIG_FILE` of Pruning here. The commands of the **subnet retraining** are as follows.

#### Subnet Retraining

```Python
python tools/train.py ${CONFIG_FILE} --cfg-options model._channel_cfg_paths=${CHANNEL_CFG_PATH} [optional arguments]
```

Different from NAS, the argument that needs to be specified here is `channel_cfg_paths` .

- `CHANNEL_CFG_PATH`: Path of `_channel_cfg_path`. `channel_cfg` represents **config for channel of the subnet searched out**, used to specify different subnets for testing.

For example, the default `_channel_cfg_paths` is set in the config below.

```Python
python ./tools/train.py \
  configs/pruning/mmcls/autoslim/autoslim_mbv2_1.5x_subnet_8xb256_in1k_flops-530M.py \
  --work-dir your_work_dir
```

### Distillation

There is only one step to start knowledge distillation.

```Python
python tools/train.py ${CONFIG_FILE} [optional arguments]
```

For example,

```Python
python ./tools/train.py \
  configs/distill/mmcls/kd/kd_logits_r34_r18_8xb32_in1k.py \
  --work-dir your_work_dir
```
