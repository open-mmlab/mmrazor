_base_ = ['./nsganetv2_mobilenet_supernet_8xb128_in1k.py']

model = dict(norm_training=True)

train_cfg = dict(
    _delete_=True,
    type='mmrazor.NSGA2SearchLoop',
    dataloader=_base_.val_dataloader,
    evaluator=_base_.val_evaluator,
    max_epochs=4,
    num_candidates=4,
    top_k=2,
    num_mutation=2,
    num_crossover=2,
    mutate_prob=0.1,
    constraints_range=dict(flops=(0., 360.)),
    score_key='accuracy/top1',
    predictor_cfg=dict(
        type='mmrazor.MetricPredictor',
        encoding_type='normal',
        train_samples=2,
        handler_cfg=dict(type='mmrazor.GaussProcessHandler')),
    finetune_cfg=dict(
        model=_base_.model,
        train_dataloader=_base_.train_dataloader,
        train_cfg=dict(by_epoch=True, max_epochs=1),
        optim_wrapper=dict(
            type='OptimWrapper',
            optimizer=dict(
                type='SGD',
                lr=0.025,
                momentum=0.9,
                weight_decay=3e-4,
                nesterov=True)),
        param_scheduler=_base_.param_scheduler,
        default_hooks=_base_.default_hooks,
    ),
)
