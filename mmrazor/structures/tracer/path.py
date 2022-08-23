# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


def _merge_node_parents(node2parents, _node2parents):
    for node, parents in _node2parents.items():
        if node in node2parents:
            cur_parents = node2parents[node]
            new_parents_set = set(cur_parents + parents)
            new_parents = list(new_parents_set)
            node2parents[node] = new_parents
        else:
            node2parents[node] = parents


class PathNode:
    """``Node`` is the data structure that represents individual instances
    within a ``Path``. It corresponds to a module or an operation such as
    concatenation in the model.

    Args:
        name (str): Unique identifier of a node.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def get_module_names(self) -> List:
        return [self.name]

    @property
    def name(self) -> str:
        """Get the name of current node."""
        return self._name

    def _get_class_name(self):
        return self.__class__.__name__

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.name == other.name
        else:
            return False

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'{self._get_class_name()}(\'{self.name}\')'


class PathConvNode(PathNode):
    """A `ConvNode` corresponds to a Conv module in the original model."""
    pass


class PathDepthWiseConvNode(PathNode):
    """A `DepthWiseConvNode` corresponds to a depth-wise conv module in the
    original model."""
    pass


class PathNormNode(PathNode):
    """A `NormNode` corresponds to a normalization module in the original
    model."""
    pass


class PathLinearNode(PathNode):
    """A `LinearNode` corresponds to a linear module in the original model."""
    pass


class Path:
    """``Path`` is the data structure that represents a list of ``Node`` traced
    by a tracer.

    Args:
        nodes(:obj:`Node` or List[:obj:`Node`], optional): Nodes in a path.
            Default to None.
    """

    def __init__(self,
                 nodes: Optional[Union[PathNode, List[PathNode]]] = None):
        self._nodes: List[PathNode] = list()
        if nodes is not None:
            if isinstance(nodes, PathNode):
                nodes = [nodes]
            assert isinstance(nodes, (list, tuple))
            for node in nodes:
                assert isinstance(node, PathNode)
                self._nodes.append(node)

    def get_root_names(self) -> List[str]:
        """Get the name of the first node in a path."""
        return self._nodes[0].get_module_names()

    def find_nodes_parents(self,
                           target_nodes: Tuple,
                           non_pass: Optional[Tuple] = None) -> Dict:
        """Find the parents of a specific node.

        Args:
            target_nodes (Tuple): Find the parents of nodes whose types
                are one of `target_nodes`.
            non_pass (Tuple): Ancestor nodes whose types are one of
                `non_pass` are the parents of a specific node. Default to None.
        """
        node2parents: Dict[str, List[PathNode]] = dict()
        for i, node in enumerate(self._nodes):
            if isinstance(node, PathConcatNode):
                _node2parents: Dict[str,
                                    List[PathNode]] = node.find_nodes_parents(
                                        target_nodes, non_pass)
                _merge_node_parents(node2parents, _node2parents)
                continue

            if isinstance(node, target_nodes):
                parents = list()
                for behind_node in self._nodes[i + 1:]:
                    if non_pass is None or isinstance(behind_node, non_pass):
                        parents.append(behind_node)
                        break
                _node2parents = {node.name: parents}
                _merge_node_parents(node2parents, _node2parents)
        return node2parents

    @property
    def nodes(self) -> List:
        """Return a list of nodes in the current path."""
        return self._nodes

    def append(self, x: PathNode) -> None:
        """Add a node to the end of the current path."""
        assert isinstance(x, PathNode)
        self._nodes.append(x)

    def pop(self, *args, **kwargs):
        """Temoves the node at the given index from the path and returns the
        removed node."""
        return self._nodes.pop(*args, **kwargs)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.nodes == other.nodes
        else:
            return False

    def __len__(self):
        return len(self._nodes)

    def __getitem__(self, item):
        return self._nodes[item]

    def __iter__(self):
        for node in self._nodes:
            yield node

    def _get_class_name(self) -> str:
        """Get the name of the current class."""
        return self.__class__.__name__

    def __repr__(self):
        child_lines = []
        for node in self._nodes:
            node_str = repr(node)
            node_str = _addindent(node_str, 2)
            child_lines.append(node_str)
        lines = child_lines

        main_str = self._get_class_name() + '('
        if lines:
            main_str += '\n  ' + ',\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str


class PathList:
    """``PathList`` is the data structure that represents a list of ``Path``
    traced by a tracer.

    Args:
        paths(:obj:`Path` or List[:obj:`Path`], optional): A list of `Path`.
            Default to None.
    """

    def __init__(self, paths: Optional[Union[Path, List[Path]]] = None):
        self._paths = list()
        if paths is not None:
            if isinstance(paths, Path):
                paths = [paths]
            assert isinstance(paths, (list, tuple))
            for path in paths:
                assert isinstance(path, Path)
                self._paths.append(path)

    def get_root_names(self) -> List[str]:
        """Get the root node of all the paths in `PathList`."""
        root_name_list = [path.get_root_names() for path in self._paths]
        for root_names in root_name_list[1:]:
            assert root_names == root_name_list[0], \
                f'If the input of a module is a concatenation of several ' \
                f'modules\' outputs, we can use `get_root_names` to get the' \
                f' names of these modules. As `get_root_names` is only used' \
                f' in this case, each element in `root_name_list` should be' \
                f' the same. Got root_name_list = {root_name_list}'
        return self._paths[0].get_root_names()

    def find_nodes_parents(self,
                           target_nodes: Tuple,
                           non_pass: Optional[Tuple] = None):
        """Find the parents of a specific node.

        Args:
            target_nodes (Tuple): Find the parents of nodes whose types
                are one of `target_nodes`.
            non_pass (Tuple): Ancestor nodes whose types are one of
                `non_pass` are the parents of a specific node. Default to None.
        """
        node2parents: Dict[str, List[PathNode]] = dict()
        for p in self._paths:
            _node2parents = p.find_nodes_parents(target_nodes, non_pass)
            _merge_node_parents(node2parents, _node2parents)
        return node2parents

    def append(self, x: Path) -> None:
        """Add a path to the end of the current PathList."""
        assert isinstance(x, Path)
        self._paths.append(x)

    @property
    def paths(self):
        """Return all paths in the current PathList."""
        return self._paths

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.paths == other.paths
        else:
            return False

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, item):
        return self._paths[item]

    def __iter__(self):
        for path in self._paths:
            yield path

    def _get_class_name(self) -> str:
        """Get the name of the current class."""
        return self.__class__.__name__

    def __repr__(self):
        child_lines = []
        for node in self._paths:
            node_str = repr(node)
            node_str = _addindent(node_str, 2)
            child_lines.append(node_str)
        lines = child_lines

        main_str = self._get_class_name() + '('
        if lines:
            main_str += '\n  ' + ',\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str


class PathConcatNode(PathNode):
    """``ConcatNode`` is the data structure that represents the concatenation
    operation in a model.

    Args:
        name (str): Unique identifier of a `ConcatNode`.
        path_lists (List[PathList]): Several nodes are concatenated and each
            node is the root node of all the paths in a `PathList`
            (one of `path_lists`).
    """

    def __init__(self, name: str, path_lists: List[PathList]):
        super().__init__(name)
        self._path_lists = list()
        for path_list in path_lists:
            assert isinstance(path_list, PathList)
            self._path_lists.append(path_list)

    def get_module_names(self) -> List[str]:
        """Several nodes are concatenated.

        Get the names of these nodes.
        """
        module_names = list()
        for path_list in self._path_lists:
            module_names.extend(path_list.get_root_names())
        return module_names

    def find_nodes_parents(self,
                           target_nodes: Tuple,
                           non_pass: Optional[Tuple] = None):
        """Find the parents of a specific node.

        Args:
            target_nodes (Tuple): Find the parents of nodes whose types
                are one of `target_nodes`.
            non_pass (Tuple): Ancestor nodes whose types are one of
                `non_pass` are the parents of a specific node. Default to None.
        """
        node2parents: Dict[str, List[PathNode]] = dict()
        for p in self._path_lists:
            _node2parents = p.find_nodes_parents(target_nodes, non_pass)
            _merge_node_parents(node2parents, _node2parents)
        return node2parents

    @property
    def path_lists(self) -> List[PathList]:
        """Return all the path_list."""
        return self._path_lists

    def __len__(self):
        return len(self._path_lists)

    def __getitem__(self, item):
        return self._path_lists[item]

    def __iter__(self):
        for path_list in self._path_lists:
            yield path_list

    def _get_class_name(self) -> str:
        """Get the name of the current class."""
        return self.__class__.__name__

    def __repr__(self):
        child_lines = []
        for node in self._path_lists:
            node_str = repr(node)
            node_str = _addindent(node_str, 2)
            child_lines.append(node_str)
        lines = child_lines

        main_str = self._get_class_name() + '('
        if lines:
            main_str += '\n  ' + ',\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str
