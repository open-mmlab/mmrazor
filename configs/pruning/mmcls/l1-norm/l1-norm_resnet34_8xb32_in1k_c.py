_base_ = ['./l1-norm_resnet34_8xb32_in1k_a.py']

un_prune = 1.0

# the config template of target_pruning_ratio can be got by
# python ./tools/get_channel_units.py {config_file} --choice
target_pruning_ratio = {
    # stage 1
    'backbone.conv1_(0, 64)_64': un_prune,  # short cut layers
    'backbone.layer1.0.conv1_(0, 64)_64': un_prune,
    'backbone.layer1.1.conv1_(0, 64)_64': un_prune,
    'backbone.layer1.2.conv1_(0, 64)_64': un_prune,
    # stage 2
    'backbone.layer2.0.conv1_(0, 128)_128': un_prune,
    'backbone.layer2.0.conv2_(0, 128)_128': un_prune,  # short cut layers
    'backbone.layer2.1.conv1_(0, 128)_128': un_prune,
    'backbone.layer2.2.conv1_(0, 128)_128': un_prune,
    'backbone.layer2.3.conv1_(0, 128)_128': un_prune,
    # stage 3
    'backbone.layer3.0.conv1_(0, 256)_256': un_prune,
    'backbone.layer3.0.conv2_(0, 256)_256': 0.8,  # short cut layers
    'backbone.layer3.1.conv1_(0, 256)_256': un_prune,
    'backbone.layer3.2.conv1_(0, 256)_256': un_prune,
    'backbone.layer3.3.conv1_(0, 256)_256': un_prune,
    'backbone.layer3.4.conv1_(0, 256)_256': un_prune,
    'backbone.layer3.5.conv1_(0, 256)_256': un_prune,
    # stage 4
    'backbone.layer4.0.conv1_(0, 512)_512': un_prune,
    'backbone.layer4.0.conv2_(0, 512)_512': un_prune,  # short cut layers
    'backbone.layer4.1.conv1_(0, 512)_512': un_prune,
    'backbone.layer4.2.conv1_(0, 512)_512': un_prune
}

model = dict(target_pruning_ratio=target_pruning_ratio, )
