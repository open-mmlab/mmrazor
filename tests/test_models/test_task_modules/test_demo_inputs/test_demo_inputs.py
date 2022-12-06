# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from ....data.tracer_passed_models import FxPassedModelManager


class TestDemoInputs(unittest.TestCase):

    def test_demo_inputs(self):
        for Model in FxPassedModelManager().include_models():
            with self.subTest(model=Model):
                demo_input = DefaultDemoInput(input_shape=[1, 3, 224, 224])
                model = Model()
                model.eval()
                try:
                    demo_input(model)
                    input = demo_input.get_data(model)
                    if isinstance(input, dict):
                        model(**input)
                    else:
                        model(input)
                except Exception as e:
                    self.fail(f'{e}')
