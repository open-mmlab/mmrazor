_base_ = ['./spos_supernet_mb_proxyless_gpu_8xb128_in1k.py']

data = dict(
    samples_per_gpu=512,
    workers_per_gpu=16,
)

algorithm = dict(bn_training_mode=True)

searcher = dict(
    type='EvolutionSearcher',
    candidate_pool_size=50,
    candidate_top_k=10,
    constraints=dict(flops=465 * 1e6),
    metrics='accuracy',
    score_key='accuracy_top-1',
    max_epoch=20,
    num_mutation=25,
    num_crossover=25,
    mutate_prob=0.1)
