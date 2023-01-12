_base_ = ['./nsganetv2_mobilenet_supernet_1xb96_cifar10.py']

model = dict(norm_training=True)

train_dataloader = dict(batch_size=256)
val_dataloader = dict(batch_size=256)
test_dataloader = val_dataloader

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
    flops_range=(0., 330.),
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
