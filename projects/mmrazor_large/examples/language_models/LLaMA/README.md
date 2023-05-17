# Examples for LLaMA

## SparseGPT

For more details about SparseGPT, please refer to [SparseGPT](../../../algorithms/SparseGPT.md)

### Usage

```shell
# example for decapoda-research/llama-7b-hf
python projects/mmrazor_large/examples/language_models/LLaMA/llama_sparse_gpt.py decapoda-research/llama-7b-hf c4

# help
usage: llama_sparse_gpt.py [-h] [--seed SEED] [--nsamples NSAMPLES] [--batch_size BATCH_SIZE] [--save SAVE] [-m M] model {wikitext2,ptb,c4}

positional arguments:
  model                 Llama model to load
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
# example for decapoda-research/llama-7b-hf
python projects/mmrazor_large/examples/language_models/LLaMA/llama_gptq.py decapoda-research/llama-7b-hf c4

# help
usage: llama_gptq.py [-h] [--seed SEED] [--nsamples NSAMPLES] [--batch_size BATCH_SIZE] [--save SAVE] [-m M] model {wikitext2,ptb,c4}

positional arguments:
  model                 Llama model to load
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
