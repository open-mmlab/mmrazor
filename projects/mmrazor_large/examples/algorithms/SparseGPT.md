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

## init sparse gpt mutator and prepare for pruning
mutator = sparse_gpt.SparseGptMutator()
mutator.prepare_from_supernet(model)

## init hessian  matrix
mutator.start_init_hessian()
infer(model, test_loader, num_samples=num_samples)
mutator.end_init_hessian()

## prune
mutator.prune_24()

## to a normal torch model
model = mutator.to_static_model(model)

```

## Full Examples

- [ResNet](../model_examples/ResNet/sparse_gpt/README.md)
- [OPT](../model_examples/language_models/OPT/README.md)
- [Llama](../model_examples/language_models/Llama/README.md)

## Cite

```latex
@article{frantar2023massive,
  title={Massive Language Models Can Be Accurately Pruned in One-Shot},
  author={Frantar, Elias and Alistarh, Dan},
  journal={arXiv preprint arXiv:2301.00774},
  year={2023}
}
```
