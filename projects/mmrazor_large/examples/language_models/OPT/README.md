# Examples for OPT

## SparseGPT

For more details about SparseGPT, please refer to [SparseGPT](../../../algorithms/SparseGPT.md)

### Usage

```shell
# example for facebook/opt-125m
python projects/mmrazor_large/examples/language_models/OPT/opt_sparse_gpt.py facebook/opt-125m c4

# help
usage: opt_sparse_gpt.py [-h] [--seed SEED] [--nsamples NSAMPLES] [--batch_size BATCH_SIZE] [--save SAVE] [-m M] model {wikitext2,ptb,c4}

positional arguments:
  model                 OPT model to load; pass `facebook/opt-X`.
  {wikitext2,ptb,c4}    Where to extract calibration data from.

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Seed for sampling the calibration data.
  --nsamples NSAMPLES   Number of calibration data samples.
  --batch_size BATCH_SIZE
                        Batchsize for calibration and evaluation.
  --save SAVE           Path to saved model.
  -m M                  Whether to enable memory efficient forward
```

## GPTQ

For more details about GPTQ, please refer to [GPTQ](../../../algorithms/GPTQ.md)

### Usage

```shell
# example for facebook/opt-125m
python projects/mmrazor_large/examples/language_models/OPT/opt_gptq.py facebook/opt-125m c4

# help
usage: opt_gptq.py [-h] [--seed SEED] [--nsamples NSAMPLES] [--batch_size BATCH_SIZE] [--save SAVE] [-m M] model {wikitext2,ptb,c4}

positional arguments:
  model                 OPT model to load; pass `facebook/opt-X`.
  {wikitext2,ptb,c4}    Where to extract calibration data from.

optional arguments:
  -h, --help            show this help message and exit
  --seed SEED           Seed for sampling the calibration data.
  --nsamples NSAMPLES   Number of calibration data samples.
  --batch_size BATCH_SIZE
                        Batchsize for calibration and evaluation.
  --save SAVE           Path to saved model.
  -m M                  Whether to enable memory efficient forward
```
