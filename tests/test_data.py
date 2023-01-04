# Copyright (c) OpenMMLab. All rights reserved.
import os
import unittest

import torch

from .data.model_library import (DefaultModelLibrary, MMClsModelLibrary,
                                 MMDetModelLibrary, MMModelLibrary,
                                 MMPoseModelLibrary, MMSegModelLibrary,
                                 ModelGenerator, TorchModelLibrary)
from .data.models import SingleLineModel
from .data.tracer_passed_models import (BackwardPassedModelManager,
                                        FxPassedModelManager)

TEST_DATA = os.getenv('TEST_DATA') == 'true'


class TestModelLibrary(unittest.TestCase):

    def test_mmcls(self):
        if not TEST_DATA:
            self.skipTest('not test data to save time.')
        library = MMClsModelLibrary(exclude=['cutmax', 'cifar'])
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_defaul_library(self):
        if not TEST_DATA:
            self.skipTest('not test data to save time.')
        library = DefaultModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_torchlibrary(self):
        if not TEST_DATA:
            self.skipTest('not test data to save time.')
        library = TorchModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_mmdet(self):
        if not TEST_DATA:
            self.skipTest('not test data to save time.')
        library = MMDetModelLibrary()
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_mmseg(self):
        if not TEST_DATA:
            self.skipTest('not test data to save time.')
        library = MMSegModelLibrary()
        print(library.short_names())

        self.assertTrue(library.is_default_includes_cover_all_models())

    # New
    def test_mmpose(self):
        if not TEST_DATA:
            self.skipTest('not test data to save time.')
        library = MMPoseModelLibrary()
        print(library.short_names())
        self.assertTrue(library.is_default_includes_cover_all_models())

    def test_get_model_by_config(self):
        config = 'mmcls::resnet/resnet34_8xb32_in1k.py'
        Model = MMModelLibrary.get_model_from_path(config)
        _ = Model()

    def test_passed_models(self):
        try:
            print(FxPassedModelManager().include_models())
            print(BackwardPassedModelManager().include_models())
        except Exception:
            self.fail()


class TestModels(unittest.TestCase):

    def _test_a_model(self, Model):
        model = Model()
        x = torch.rand(2, 3, 224, 224)
        y = model(x)
        self.assertSequenceEqual(y.shape, [2, 1000])

    def test_models(self):
        library = DefaultModelLibrary()
        for Model in library.include_models():
            with self.subTest(model=Model):
                self._test_a_model(Model)

    def test_generator(self):
        Model = ModelGenerator('model', SingleLineModel)
        model = Model()
        model.eval()
        self.assertEqual(model.training, False)
        model.train()
        self.assertEqual(model.training, True)
