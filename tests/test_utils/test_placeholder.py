# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import pytest

from mmrazor.utils import get_placeholder


class TestPlaceholder(unittest.TestCase):

    def test_placeholder(self):
        holder = get_placeholder('test')
        with pytest.raises(ImportError):
            holder()
        with pytest.raises(ImportError):
            holder.a
        with pytest.raises(ImportError):
            holder.a()
