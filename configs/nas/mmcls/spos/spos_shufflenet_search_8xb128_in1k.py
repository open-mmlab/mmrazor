_base_ = ['./spos_shufflenet_supernet_8xb128_in1k.py']

model = dict(norm_training=True)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=20,
    num_candidates=50,
    top_k=10,
    num_mutation=25,
    num_crossover=25,
    mutate_prob=0.1,
    flops_range=(0., 330.),
    score_key='accuracy/top1')
