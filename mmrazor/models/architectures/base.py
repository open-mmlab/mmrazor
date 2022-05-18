# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import BaseModule

from mmrazor.registry import MODELS


class BaseArchitecture(BaseModule):
    """Base class for architecture.

    Args:
        model (:obj:`torch.nn.Module`): Model to be slimmed, such as
            ``DETECTOR`` in MMDetection.
    """

    def __init__(self, model, **kwargs):
        super(BaseArchitecture, self).__init__(**kwargs)
        self.model = MODELS.build(model)

    # TODO maybe depercated.
    def forward_dummy(self, *args, **kwargs):
        """Used for calculating network flops."""
        assert hasattr(self.model, 'forward_dummy')
        return self.model.forward_dummy(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        return self.model(*args, **kwargs)

    def show_result(self, *args, **kwargs):
        """Draw `result` over `img`"""
        return self.model.show_result(*args, **kwargs)
