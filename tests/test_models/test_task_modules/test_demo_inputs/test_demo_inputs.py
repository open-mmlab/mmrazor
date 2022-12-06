# Copyright (c) OpenMMLab. All rights reserved.
import unittest

from mmrazor.models.task_modules.demo_inputs import DefaultDemoInput
from ....data.tracer_passed_models import BackwardPassedModelManager


class TestDemoInputs(unittest.TestCase):

    def test_demo_inputs(self):
        for Model in BackwardPassedModelManager().include_models():
            with self.subTest(model=Model):
                demo_input = DefaultDemoInput()
                model = Model()
                try:
                    demo_input(model)
                    input = demo_input.get_data(model)
                    if isinstance(input, dict):
                        model(**input)
                    else:
                        model(input)
                except Exception as e:
                    self.fail(f'{e}')
