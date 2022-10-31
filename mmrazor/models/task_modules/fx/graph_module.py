# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Union

import torch
import torch.fx
from mmengine.model import BaseModel
from torch.fx import Graph
from torch.fx.graph_module import GraphModule, _copy_attr


class MMGraphModule(GraphModule, BaseModel):
    _graph_map: Dict[str, Graph] = dict()

    def __init__(self, root, graph, class_name) -> None:
        self.data_preprocessor = root.data_preprocessor
        if isinstance(graph, Graph):
            super().__init__(root, graph, class_name)

        elif isinstance(graph, (dict)):
            assert len(graph) > 0, '`graph` should has 1 Graph at least.'

            self._custom_init(root, graph, class_name)

    def _custom_init(self, root, graphs, class_name):
        self._graph_map = graphs

        super(GraphModule, self).__init__()
        self.__class__.__name__ = class_name
        if isinstance(root, torch.nn.Module):
            if hasattr(root, 'training'):
                self.training = root.training
            for _, graph in graphs.items():
                for node in graph.nodes:
                    if node.op in ['get_attr', 'call_module']:
                        assert isinstance(node.target, str)
                        _copy_attr(root, self, node.target)
        elif isinstance(root, dict):
            raise NotImplementedError(
                'Customed GraphModule do not support `root` of `dict` type')
        else:
            raise RuntimeError('Unsupported type ' + str(root) +
                               ' passed for root!')

        # Set the first graph as default graph.
        self.graph = list(graphs.values())[0]

        # Store the Tracer class responsible for creating a Graph separately
        # as part of the GraphModule state, except when the Tracer is defined
        # in a local namespace. Locally defined Tracers are not pickleable.
        # This is needed because torch.package will serialize a GraphModule
        # without retaining the Graph, and needs to use the correct Tracer
        # to re-create the Graph during deserialization.
        self._tracer_cls = None
        if self.graph._tracer_cls and (
                '<locals>' not in self.graph._tracer_cls.__qualname__):
            self._tracer_cls = self.graph._tracer_cls

        self._tracer_extras = {}
        if self.graph._tracer_extras:
            self._tracer_extras = self.graph._tracer_extras

        # Dictionary to store metadata
        self.meta: Dict[str, Any] = {}

    def to_mode(self, mode):
        graph = self._graph_map.get(mode, None)
        if graph:
            self.graph = graph
            self.recompile()
        else:
            raise KeyError(
                f'`self._graph_map` has no graph named `{mode}`, expecting one of: {list(self._graph_map.keys())}'  # noqa: E501
            )

    def _run_forward(self, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`
        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.
        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            self.to_mode(mode)
            results = self(**data)
        elif isinstance(data, (list, tuple)):
            self.to_mode(mode)
            results = self(*data)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results
