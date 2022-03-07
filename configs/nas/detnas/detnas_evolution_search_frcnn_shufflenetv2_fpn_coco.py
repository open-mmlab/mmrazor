_base_ = ['./detnas_supernet_frcnn_shufflenetv2_fpn_1x_coco.py']

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
)

algorithm = dict(bn_training_mode=False)

searcher = dict(
    type='EvolutionSearcher',
    metrics='bbox',
    score_key='bbox_mAP',
    constraints=dict(flops=300 * 1e6),
    candidate_pool_size=50,
    candidate_top_k=10,
    max_epoch=20,
    num_mutation=20,
    num_crossover=20,
)
