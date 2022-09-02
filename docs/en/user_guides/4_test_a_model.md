# Test a model

### NAS

To test nas method, you can use the following command.

```Python
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --cfg-options algorithm.fix_subnet=${FIX_SUBNET_PATH} [optional arguments]
```

- `FIX_SUBNET_PATH`: Path of `fix_subnet`. `fix_subnet` represents **config for mutable of the subnet searched out**, used to specify different subnets for testing. An example for `fix_subnet` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml).

The usage of optional arguments are the same as corresponding tasks like mmclassification, mmdetection and mmsegmentation.

For example,

```Python
python tools/test.py \
  configs/nas/mmcls/spos/spos_subnet_shufflenetv2_8xb128_in1k.py \
  your_subnet_checkpoint_path \
  --cfg-options algorithm.fix_subnet=configs/nas/mmcls/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml
```

### Pruning

#### Split Checkpoint(Optional)

If you train a slimmable model during retraining, checkpoints of different subnets are actually fused in only one checkpoint. You can split this checkpoint to multiple independent checkpoints by using the following command

```Python
python tools/model_converters/split_checkpoint.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --channel-cfgs ${CHANNEL_CFG_PATH} [optional arguments]
```

- `CHANNEL_CFG_PATH`: A list of paths of `channel_cfg`. For example, when you retrain a slimmable model, your command will be like `--cfg-options algorithm.channel_cfg=cfg1,cfg2,cfg3`. And your command here should be `--channel-cfgs cfg1 cfg2 cfg3`. The order of them should be the same.

For example,

```Python
python tools/model_converters/split_checkpoint.py \
  configs/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k.py \
  your_retraining_checkpoint_path \
  --channel-cfgs configs/pruning/autoslim/AUTOSLIM_MBV2_530M_OFFICIAL.yaml configs/pruning/autoslim/AUTOSLIM_MBV2_320M_OFFICIAL.yaml configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml
```

#### Test

To test pruning method, you can use following command

```Python
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --cfg-options model._channel_cfg_paths=${CHANNEL_CFG_PATH} [optional arguments]
```

- `task`: one of `mmcls`、`mmdet` and `mmseg`

- `CHANNEL_CFG_PATH`: Path of `channel_cfg`. `channel_cfg` represents **config for channel of the subnet searched out**, used to specify different subnets for testing. An example for `channel_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml), and the usage can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/README.md#test-a-subnet).

For example,

```Python
python ./tools/test.py \
  configs/pruning/mmcls/autoslim/autoslim_mbv2__1.5x_subnet_8xb256_in1k-530M.py \
  your_splitted_checkpoint_path --metrics accuracy
```

### Distillation

To test the distillation method, you can use the following command

```Python
python tools/test.py ${CONFIG_FILE} ${CHECKPOINT_PATH} [optional arguments]
```

For example,

```Python
python ./tools/test.py \
  configs/distill/mmseg/cwd/cwd_logits_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k.py \
  your_splitted_checkpoint_path --show
```
