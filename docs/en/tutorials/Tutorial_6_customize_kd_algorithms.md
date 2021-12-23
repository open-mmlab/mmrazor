# Toturial 6: Customize KD algorithms

Here we show how to develop new KD algorithms with an example of cwd.

1. Register a new algorithm

    Create a new file `mmrazor/models/algorithms/kd.py`, class `Distillation` inherits from class `BaseAlgorithm`

    ```python
    from mmrazor.models.builder import ALGORITHMS
    from .base import BaseAlgorithm

    @ALGORITHMS.register_module()
    class Distillation(BaseAlgorithm):
        def __init__(self, use_gt, **kwargs):
            super(Distillation, self).__init__(**kwargs)
            self.use_gt = use_gt
            pass

        def train_step(self, data, optimizer):
            pass
    ```

2. Develop new algorithm components (optional)

    Distillation can directly use class `SingleTeacherDistiller` or other distillers in `mmrazor/models/distillers/` as core functions provider. If distillers provided in MMRazor don't meet your needs, you can develop new algorithm components for your algorithm as follows:

    a. Create a new file `mmrazor/models/distillers/multi_teachers.py`, class `MultiTeachersDistiller` inherits from class `SingleTeacherDistiller`

    b. Finish the functions you need, eg: `build_teacher`, `compute_distill_loss` and so on

    ```python
    from mmrazor.models.builder import DISTILLERS
    from .single_teacher import SingleTeacherDistiller

    @DISTILLERS.register_module()
    class MultiTeachersDistiller(SingleTeacherDistiller):
        def __init__(self, teacher, teacher_ckpt, teacher_trainable, **kwargs):
            super(MultiTeachersDistiller,
                self).__init__(teacher, teacher_trainable, **kwargs)

        def build_teacher(self, cfgs, ckpts):
            pass

        def exec_teacher_forward(self, data, return_output):
            pass

        def compute_distill_loss(self):
            pass
    ```

   c. Import the module in `mmrazor/models/distillers/__init__.py`

    ```python
    from .multi_teachers import MultiTeachersDistiller

    __all__ = [..., 'MultiTeachersDistiller']
    ```

3. Rewrite its train_step

    Develop key logic of your algorithm in function`train_step`

    ```python
    from mmrazor.models.builder import ALGORITHMS
    from .base import BaseAlgorithm

    @ALGORITHMS.register_module()
    class Distillation(BaseAlgorithm):
        def __init__(self, use_gt, **kwargs):
            super(Distillation, self).__init__(**kwargs)
            self.use_gt = use_gt
            pass

        def train_step(self, data, optimizer):
            losses = dict()
            if self.use_gt:
                _ = self.distiller.exec_teacher_forward(data)
                gt_losses = self.distiller.exec_student_forward(
                    self.architecture, data)
                distill_losses = self.distiller.compute_distill_loss()
                losses.update(gt_losses)
                losses.update(distill_losses)
            else:
                _ = self.distiller.exec_teacher_forward(data)
                _ = self.distiller.exec_student_forward(self.architecture, data)
                distill_losses = self.distiller.compute_distill_loss()
                losses.update(distill_losses)

            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
            return outputs
    ```

4. Add your custom functions (optional)

    After finishing your key logic in function `train_step`, if you also need other custom functions, you can add them in class `Distillation`

5. Import the class

    You can either add the following line to `mmrazor/models/algorithms/__init__.py`

    ```python
    from .kd import Distillation

    __all__ = [..., 'Distillation']
    ```

    or alternatively add

    ```python
    custom_imports = dict(
        imports=['mmrazor.models.algorithms.spos'],
        allow_failed_imports=False)
    ```

    to the config file to avoid modifying the original code.

6. Use the algorithm in your config file

    ```python
    algorithm = dict(
        type='Distillation',
        distiller=dict(type='MultiTeachersDistiller', ...),  # you can also use your new algorithm components here
        ...
    )
    ```
