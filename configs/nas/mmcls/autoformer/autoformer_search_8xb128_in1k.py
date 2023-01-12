_base_ = ['./autoformer_supernet_32xb256_in1k.py']

custom_hooks = None

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=20,
    num_candidates=20,
    top_k=10,
    num_mutation=5,
    num_crossover=5,
    mutate_prob=0.2,
    constraints_range=dict(params=(0, 55)),
    score_key='accuracy/top1')
