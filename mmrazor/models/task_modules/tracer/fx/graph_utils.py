# Copyright (c) OpenMMLab. All rights reserved.
import copy

from torch.ao.quantization.fake_quantize import FakeQuantizeBase


def _get_attrs(target, attrs):

    attrs = attrs.split('.')

    for att in attrs:
        target = getattr(target, att, None)
    return target


def del_fakequant_before_target(prepared_model, target_patterns, inplace=True):

    def recursive_find_erased_nodes(node):
        """Find FakeQuant before target node recursively.

        Examples:
            head_fc = self.head.fc(activation_post_process_87);  \
                activation_post_process_87 = None
            activation_post_process_88 = \
                self.activation_post_process_88(head_fc);  head_fc = None
            head = self.head
            _get_loss = head._get_loss(activation_post_process_88,
                data_samples);  \
                head = activation_post_process_88 = data_samples = None
            return _get_loss

        node                       |           node.args
        --------------------
        output                     | (_get_loss, )
        _get_loss                  | (head, activation_post_process_88,
                                        data_samples)
        head                       | ()
        activation_post_process_88 | (head_fc, )
        data_samples               | (None, )
        """
        if node is None:
            return
        # if node.op != 'call_module':
        #     return
        if isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            nodes_to_erase.append(node)
            return
        for prev_node in node.args:
            recursive_find_erased_nodes(prev_node)
        for prev_node in node.kwargs.values():
            recursive_find_erased_nodes(prev_node)
        return

    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if isinstance(node.target, str) and node.target in target_patterns:
            nodes_to_erase = []
            recursive_find_erased_nodes(node)
            print(node, nodes_to_erase)
            for to_erase in nodes_to_erase:
                to_erase.replace_all_uses_with(to_erase.args[0])
                new_graph.erase_node(to_erase)
                delattr(prepared_model, to_erase.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_target(prepared_model, target_patterns, inplace=True):
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if isinstance(prev_node.target, str) and (prev_node.target
                                                      not in target_patterns):
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_before_module(prepared_model, module_patterns, inplace=True):
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), module_patterns):
            to_erase = node.args[0]
            if not isinstance(
                    _get_attrs(prepared_model, to_erase.target),
                    FakeQuantizeBase):
                continue
            if len(to_erase.users) > 1:
                continue
            to_erase.replace_all_uses_with(to_erase.args[0])
            new_graph.erase_node(to_erase)
            delattr(prepared_model, to_erase.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model


def del_fakequant_after_module(prepared_model, module_patterns, inplace=True):
    if not inplace:
        prepared_model = copy.deepcopy(prepared_model)
    new_graph = copy.deepcopy(prepared_model.graph)
    for node in new_graph.nodes:
        if node.op == 'call_module' and isinstance(
                _get_attrs(prepared_model, node.target), FakeQuantizeBase):
            assert len(node.args) == 1
            prev_node = node.args[0]
            if prev_node.op != 'call_module':
                continue
            print(prev_node.target)
            if not isinstance(
                    _get_attrs(prepared_model, prev_node.target),
                    module_patterns):
                continue
            node.replace_all_uses_with(prev_node)
            new_graph.erase_node(node)
            delattr(prepared_model, node.target)
    new_graph.lint()
    prepared_model.graph = new_graph
    return prepared_model
