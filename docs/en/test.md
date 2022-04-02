# Test a model

## NAS

To test nas method, you can use following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --cfg-options algorithm.mutable_cfg=${MUTABLE_CFG_PATH} [optional arguments]
```

- `task`: one of ``mmcls``、``mmdet`` and ``mmseg``
- `MUTABLE_CFG_PATH`: Path of `mutable_cfg`. `mutable_cfg` represents **config for mutable of the subnet searched out**, used to specify different subnets for testing. An example for `mutable_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml).

The usage of optional arguments are the same as corresponding tasks like mmclassification, mmdetection and mmsegmentation.

For example,

<pre>
python tools/mmcls/test_mmcls.py \
  configs/nas/spos/spos_subnet_shufflenetv2_8xb128_in1k.py \
  <em>your_subnet_checkpoint_path</em> \
  --cfg-options algorithm.mutable_cfg=configs/nas/spos/SPOS_SHUFFLENETV2_330M_IN1k_PAPER.yaml
</pre>

## Pruning

### Split Checkpoint(Optional)
If you train a slimmable model during retraining, checkpoints of different subnets are
actually fused in only one checkpoint. You can split this checkpoint to
multiple independent checkpoints by using the following command

```bash
python tools/model_converters/split_checkpoint.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --channel-cfgs ${CHANNEL_CFG_PATH} [optional arguments]
```

- `CHANNEL_CFG_PATH`: A list of paths of `channel_cfg`. For example, when you
retrain a slimmable model, your command will be like `--cfg-options algorithm.channel_cfg=cfg1,cfg2,cfg3`.
And your command here should be `--channel-cfgs cfg1 cfg2 cfg3`. The order of them should be the same.

For example,

<pre>
python tools/model_converters/split_checkpoint.py \
  configs/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k.py \
  <em>your_retraining_checkpoint_path</em> \
  --channel-cfgs configs/pruning/autoslim/AUTOSLIM_MBV2_530M_OFFICIAL.yaml configs/pruning/autoslim/AUTOSLIM_MBV2_320M_OFFICIAL.yaml configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml
</pre>

### Test

To test pruning method, you can use following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} --cfg-options algorithm.channel_cfg=${CHANNEL_CFG_PATH} [optional arguments]
```

- `task`: one of ``mmcls``、``mmdet`` and ``mmseg``
- `CHANNEL_CFG_PATH`: Path of `channel_cfg`. `channel_cfg` represents **config for channel of the subnet searched out**, used to specify different subnets for testing. An example for `channel_cfg` can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/AUTOSLIM_MBV2_220M_OFFICIAL.yaml), and the usage can be found [here](https://github.com/open-mmlab/mmrazor/blob/master/configs/pruning/autoslim/README.md#test-a-subnet).

For example,

<pre>
python ./tools/mmcls/test_mmcls.py \
  configs/pruning/autoslim/autoslim_mbv2_subnet_8xb256_in1k.py \
  <em>your_splitted_checkpoint_path</em> --metrics accuracy \
  --cfg-options algorithm.channel_cfg=configs/pruning/autoslim/AUTOSLIM_MBV2_530M_OFFICIAL.yaml
</pre>


## Distillation

To test distillation method, you can use the following command

```bash
python tools/${task}/test_${task}.py ${CONFIG_FILE} ${CHECKPOINT_PATH} [optional arguments]
```

- `task`: one of ``mmcls``、``mmdet`` and ``mmseg``

For example,

<pre>
python ./tools/mmseg/test_mmseg.py \
  configs/distill/cwd/cwd_cls_head_pspnet_r101_d8_pspnet_r18_d8_512x1024_cityscapes_80k.py \
  <em>your_splitted_checkpoint_path</em> --show
</pre>
