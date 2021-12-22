# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta

from mmcv.runner import BaseModule

from mmrazor.models.architectures import Placeholder
from mmrazor.models.builder import MUTABLES, MUTATORS
from mmrazor.models.mutables import MutableModule


@MUTATORS.register_module()
class BaseMutator(BaseModule, metaclass=ABCMeta):
    """Base class for mutators."""

    def __init__(self, placeholder_mapping=None, init_cfg=None):
        super(BaseMutator, self).__init__(init_cfg=init_cfg)
        self.placeholder_mapping = placeholder_mapping

    def prepare_from_supernet(self, supernet):
        """Implement some preparatory work based on supernet, including
        ``convert_placeholder`` and ``build_search_spaces``.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.
        """
        if self.placeholder_mapping is not None:
            self.convert_placeholder(supernet, self.placeholder_mapping)

        self.search_spaces = self.build_search_spaces(supernet)

    def build_search_spaces(self, supernet):
        """Build a search space from the supernet.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used
                in your algorithm.

        Returns:
            dict: To collect some information about ``MutableModule`` in the
                supernet.
        """
        search_spaces = dict()

        def traverse(module):
            for child in module.children():
                if isinstance(child, MutableModule):
                    if child.space_id not in search_spaces.keys():
                        search_spaces[child.space_id] = dict(
                            modules=[child],
                            choice_names=child.choice_names,
                            num_chosen=child.num_chosen,
                            space_mask=child.build_space_mask())
                    else:
                        search_spaces[child.space_id]['modules'].append(child)

                traverse(child)

        traverse(supernet)
        return search_spaces

    def convert_placeholder(self, supernet, placeholder_mapping):
        """Replace all placeholders in the model.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used in
                your algorithm.
            placeholder_mapping (dict): Record which placeholders need to be
                replaced by which ops,
                its keys are the properties ``placeholder_group`` of
                placeholders used in the searchable architecture,
                its values are the registered ``OPS``.
        """

        def traverse(module):

            for name, child in module.named_children():
                if isinstance(child, Placeholder):

                    mutable_cfg = placeholder_mapping[
                        child.placeholder_group].copy()
                    assert 'type' in mutable_cfg, f'{mutable_cfg}'
                    mutable_type = mutable_cfg.pop('type')
                    assert mutable_type in MUTABLES, \
                        f'{mutable_type} not in MUTABLES.'
                    mutable_constructor = MUTABLES.get(mutable_type)
                    mutable_kwargs = child.placeholder_kwargs
                    mutable_kwargs.update(mutable_cfg)
                    mutable_module = mutable_constructor(**mutable_kwargs)
                    setattr(module, name, mutable_module)

                    # setattr(module, name, choice_module)
                    # If the new MUTABLE is MutableEdge, it may have MutableOP,
                    # so here we need to traverse the new MUTABLES.
                    traverse(mutable_module)
                else:
                    traverse(child)

        traverse(supernet)

    def deploy_subnet(self, supernet, subnet_dict):
        """Export the subnet from the supernet based on the specified
        subnet_dict.

        Args:
            supernet (:obj:`torch.nn.Module`): The architecture to be used in
                your algorithm.
            subnet_dict (dict): Record the information to build the subnet from
                the supernet,
                its keys are the properties ``space_id`` of placeholders in the
                mutator's search spaces,
                its values are dicts: {'chosen': ['chosen name1',
                'chosen name2', ...]}
        """

        def traverse(module):
            for name, child in module.named_children():
                if isinstance(child, MutableModule):
                    space_id = child.space_id
                    chosen = subnet_dict[space_id]['chosen']
                    child.export(chosen)

                traverse(child)

        traverse(supernet)
