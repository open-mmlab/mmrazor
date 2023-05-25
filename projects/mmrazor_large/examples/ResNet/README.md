# Examples for ResNet

## SparseGPT

For more details about SparseGPT, please refer to [SparseGPT](../../algorithms/SparseGPT.md)

### Usage

```shell
python projects/mmrazor_large/examples/ResNet/resnet18_sparse_gpt.py --data {imagenet_path} --batchsize 128 --num_samples 512
```

**Note**: this imagenet folder follows torch format.

## GPTQ

For more details about GPTQ, please refer to [GPTQ](../../algorithms/GPTQ.md)

### Usage

```shell
python projects/mmrazor_large/examples/ResNet/resnet18_gptq.py --data {imagenet_path} --batchsize 128 --num_samples 512
```

**Note**: this imagenet folder follows torch format.
