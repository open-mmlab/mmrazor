_base_ = ['./spos_shufflenet_supernet_8xb128_in1k.py']

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=2,
    num_candidates=4,
    top_k=2,
    num_mutation=2,
    num_crossover=2,
    mutate_prob=0.1)
