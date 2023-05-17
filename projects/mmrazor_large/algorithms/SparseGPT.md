# SparseGPT

## abstract

We show for the first time that large-scale generative pretrained transformer (GPT) family models can be pruned to at least 50% sparsity in one-shot, without any retraining, at minimal loss of accuracy. This is achieved via a new pruning method called SparseGPT, specifically designed to work efficiently and accurately on massive GPT-family models. We can execute SparseGPT on the largest available open-source models, OPT-175B and BLOOM-176B, in under 4.5 hours, and can reach 60% unstructured sparsity with negligible increase in perplexity: remarkably, more than 100 billion weights from these models can be ignored at inference time. SparseGPT generalizes to semi-structured (2:4 and 4:8) patterns, and is compatible with weight quantization approaches.

## Usage

SparseGPT is easy to use in mmrazor. You can use it like this:

```python
from mmrazor.implementations.pruning import sparse_gpt

# initial model, dataloaders
model
train_loader, test_loader

## init sparse gpt compressor and prepare for pruning
compressor = sparse_gpt.SparseGptCompressor()
compressor.prepare(model)

## init hessian  matrix
compressor.register_hessian_hooks()
infer(model, test_loader, num_samples=num_samples)
compressor.remove_hessian_hooks()

## prune
compressor.prune_24()

## to a normal torch model
model = compressor.to_static_model(model)

```

## Full Examples

- [ResNet](../examples/ResNet/sparse_gpt/README.md)
- [OPT](../examples/language_models/OPT/README.md)
- [Llama](../examples/language_models/Llama/README.md)

## Cite

```latex
@article{frantar2023massive,
  title={Massive Language Models Can Be Accurately Pruned in One-Shot},
  author={Frantar, Elias and Alistarh, Dan},
  journal={arXiv preprint arXiv:2301.00774},
  year={2023}
}
```
