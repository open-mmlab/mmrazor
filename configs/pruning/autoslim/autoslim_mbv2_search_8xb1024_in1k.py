_base_ = [
    './autoslim_mbv2_supernet_8xb256_in1k.py',
]

algorithm = dict(distiller=None, input_shape=(3, 224, 224))

searcher = dict(
    type='GreedySearcher',
    target_flops=[500000000, 300000000, 200000000],
    max_channel_bins=12,
    metrics='accuracy')

data = dict(samples_per_gpu=1024, workers_per_gpu=4)
