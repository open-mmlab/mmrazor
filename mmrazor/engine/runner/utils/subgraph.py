# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch.fx as fx


def extract_subgraph(graphmodule, block_slice):
    subgraph = copy.deepcopy(graphmodule.graph)
    block_start, block_end = block_slice[:2]
    for node in subgraph.nodes:
        if node.name == 'inputs':
            input_node = node
        if node.name == block_start.name:
            node.replace_input_with(node.prev, input_node)
        if node.name == block_end.name:
            output_node = node
        if node.op == 'output':
            node.replace_input_with(node.prev, output_node)
    subgraph.lint()
    subgraph_module = fx.GraphModule(graphmodule, subgraph)
    subgraph_module.graph.eliminate_dead_code()
    subgraph_module.recompile()
    return subgraph_module


def extract_blocks(graph, key_word='layer'):
    block_slices = []
    block_slice = []
    pre_stage_index, pre_block_index = 0, 0
    cur_stage_index, cur_block_index = 0, 0
    for node in graph.nodes:
        if key_word not in node.name:
            continue
        else:
            items = node.name.split('_')
            for i, item in enumerate(items):
                if key_word in item:
                    cur_stage_index = int(item[5:])
                    cur_block_index = int(items[i + 1])
                    break
            if (cur_block_index != pre_block_index) or (cur_stage_index !=
                                                        pre_stage_index):
                block_slice.append(node.prev)
                if len(block_slice) == 2:
                    block_slices.append(block_slice)
                block_slice = []
                block_slice.append(node)

            pre_stage_index, pre_block_index = cur_stage_index, cur_block_index

    return block_slices


def extract_layers(graphmodule, layer_types):
    layer_slices = []
    for node in graphmodule.graph.nodes:
        if node.op == 'call_module':
            m = graphmodule.get_submodule(node.target)
            if isinstance(m, layer_types):
                layer_slices.append((node, node))
    return layer_slices
