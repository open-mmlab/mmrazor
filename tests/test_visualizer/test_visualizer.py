# Copyright (c) OpenMMLab. All rights reserved.
from unittest import TestCase

import numpy as np
import pytest
import torch
from mmengine.visualization import Visualizer

from mmrazor.visualization.local_visualizer import modify


class TestVisualizer(TestCase):

    def setUp(self):
        """Setup the demo image in every test method.

        TestCase calls functions in this order: setUp() -> testMethod() ->
        tearDown() -> cleanUp()
        """
        self.image = np.random.randint(
            0, 256, size=(10, 10, 3)).astype('uint8')

    def test_draw_featmap(self):
        visualizer = Visualizer()
        visualizer.draw_featmap = modify
        image = np.random.randint(0, 256, size=(3, 3, 3), dtype='uint8')

        # must be Tensor
        with pytest.raises(
                AssertionError,
                match='`featmap` should be torch.Tensor, but got '
                "<class 'numpy.ndarray'>"):
            visualizer.draw_featmap(np.ones((3, 3, 3)))

        # test tensor format
        with pytest.raises(
                AssertionError, match='Input dimension must be 3, but got 4'):
            visualizer.draw_featmap(torch.randn(1, 1, 3, 3))

        # test overlaid_image shape
        with pytest.warns(Warning):
            visualizer.draw_featmap(torch.randn(1, 4, 3), overlaid_image=image)

        # test resize_shape
        featmap = visualizer.draw_featmap(
            torch.randn(1, 4, 3), resize_shape=(6, 7))
        assert featmap.shape[:2] == (6, 7)
        featmap = visualizer.draw_featmap(
            torch.randn(1, 4, 3), overlaid_image=image, resize_shape=(6, 7))
        assert featmap.shape[:2] == (6, 7)

        # test channel_reduction parameter
        # mode only supports 'squeeze_mean' and 'select_max'
        with pytest.raises(AssertionError):
            visualizer.draw_featmap(
                torch.randn(2, 3, 3), channel_reduction='xx')

        featmap = visualizer.draw_featmap(
            torch.randn(2, 3, 3), channel_reduction='squeeze_mean')
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(2, 3, 3), channel_reduction='select_max')
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(2, 3, 3), channel_reduction='pixel_wise_max')
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(2, 4, 3),
            overlaid_image=image,
            channel_reduction='pixel_wise_max')
        assert featmap.shape[:2] == (3, 3)

        # test topk parameter
        with pytest.raises(
                AssertionError,
                match='The input tensor channel dimension must be 1 or 3 '
                'when topk is less than 1, but the channel '
                'dimension you input is 6, you can use the '
                'channel_reduction parameter or set topk '
                'greater than 0 to solve the error'):
            visualizer.draw_featmap(
                torch.randn(6, 3, 3), channel_reduction=None, topk=0)

        featmap = visualizer.draw_featmap(
            torch.randn(6, 3, 3), channel_reduction='select_max', topk=10)
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(1, 4, 3), channel_reduction=None, topk=-1)
        assert featmap.shape[:2] == (4, 3)

        featmap = visualizer.draw_featmap(
            torch.randn(3, 4, 3),
            overlaid_image=image,
            channel_reduction=None,
            topk=-1)
        assert featmap.shape[:2] == (3, 3)
        featmap = visualizer.draw_featmap(
            torch.randn(6, 3, 3),
            channel_reduction=None,
            topk=4,
            arrangement=(2, 2))
        assert featmap.shape[:2] == (6, 6)
        featmap = visualizer.draw_featmap(
            torch.randn(6, 3, 3),
            channel_reduction=None,
            topk=4,
            arrangement=(1, 4))
        assert featmap.shape[:2] == (3, 12)
        with pytest.raises(
                AssertionError,
                match='The product of row and col in the `arrangement` '
                'is less than topk, please set '
                'the `arrangement` correctly'):
            visualizer.draw_featmap(
                torch.randn(6, 3, 3),
                channel_reduction=None,
                topk=4,
                arrangement=(1, 2))

        # test gray
        featmap = visualizer.draw_featmap(
            torch.randn(6, 3, 3),
            overlaid_image=np.random.randint(
                0, 256, size=(3, 3), dtype='uint8'),
            channel_reduction=None,
            topk=4,
            arrangement=(2, 2))
        assert featmap.shape[:2] == (6, 6)
