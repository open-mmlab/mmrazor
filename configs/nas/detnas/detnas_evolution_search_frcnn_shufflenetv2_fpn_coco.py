_base_ = ['./detnas_frcnn_shufflenet_fpn_supernet_coco_1x.py']

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
)

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
