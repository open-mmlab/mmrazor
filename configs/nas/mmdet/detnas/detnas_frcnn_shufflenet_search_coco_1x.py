_base_ = ['./detnas_frcnn_shufflenet_supernet_coco_1x.py']

model = dict(norm_training=True)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.EvolutionSearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=20,
    num_candidates=50,
    top_k=10,
    num_mutation=20,
    num_crossover=20,
    mutate_prob=0.1,
    constraints_range=dict(flops=(0, 330)),
    score_key='coco/bbox_mAP')
