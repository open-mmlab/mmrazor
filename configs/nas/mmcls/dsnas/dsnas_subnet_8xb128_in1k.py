_base_ = ['./dsnas_supernet_8xb128_in1k.py']

# NOTE: Replace this with the mutable_cfg searched by yourself.
fix_subnet = {
    'backbone.layers.0.0': 'shuffle_3x3',
    'backbone.layers.0.1': 'shuffle_7x7',
    'backbone.layers.0.2': 'shuffle_3x3',
    'backbone.layers.0.3': 'shuffle_5x5',
    'backbone.layers.1.0': 'shuffle_3x3',
    'backbone.layers.1.1': 'shuffle_3x3',
    'backbone.layers.1.2': 'shuffle_3x3',
    'backbone.layers.1.3': 'shuffle_7x7',
    'backbone.layers.2.0': 'shuffle_xception',
    'backbone.layers.2.1': 'shuffle_3x3',
    'backbone.layers.2.2': 'shuffle_3x3',
    'backbone.layers.2.3': 'shuffle_5x5',
    'backbone.layers.2.4': 'shuffle_3x3',
    'backbone.layers.2.5': 'shuffle_5x5',
    'backbone.layers.2.6': 'shuffle_7x7',
    'backbone.layers.2.7': 'shuffle_7x7',
    'backbone.layers.3.0': 'shuffle_xception',
    'backbone.layers.3.1': 'shuffle_3x3',
    'backbone.layers.3.2': 'shuffle_7x7',
    'backbone.layers.3.3': 'shuffle_3x3',
}

model = dict(fix_subnet=fix_subnet)

find_unused_parameters = False
