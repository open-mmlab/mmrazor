_base_ = [
    './supernet.py',
]

algorithm = dict(input_shape=(3, 224, 224))

searcher = dict(
    type='BCNetSearcher',
    max_channel_bins=20,
    candidate_pool_size=5,
    candidate_top_k=2,
    constraints=dict(flops=200 * 1e6),
    metrics='accuracy',
    score_key='accuracy_top-1',
    max_epoch=50,
    num_mutation=2,
    num_crossover=3,
    mutate_prob=0.1
)

data = dict(samples_per_gpu=1024, workers_per_gpu=4)
