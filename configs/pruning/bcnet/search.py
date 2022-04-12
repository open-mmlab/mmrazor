_base_ = [
    './supernet.py',
]

algorithm = dict(input_shape=(3, 224, 224))

searcher = dict(
    type='BCNetSearcher',
    max_channel_bins=20,
    candidate_pool_size=50,
    candidate_top_k=10,
    constraints=dict(flops=210 * 1e6),
    metrics='accuracy',
    score_key='accuracy_top-1',
    max_epoch=40,
    num_mutation=25,
    num_crossover=25,
    mutate_prob=0.1
)

data = dict(samples_per_gpu=1024, workers_per_gpu=4)
