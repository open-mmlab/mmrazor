# Tutorial 5: Customize Pruning algorithms

Here we show how to develop new Pruning algorithms with an example of AutoSlim.

1. Register a new algorithm

    Create a new file `mmrazor/models/algorithms/autoslim.py`, class `AutoSlim` inherits from class `BaseAlgorithm`

    ```python
    from mmrazor.models.builder import ALGORITHMS
    from .base import BaseAlgorithm

    @ALGORITHMS.register_module()
    class AutoSlim(BaseAlgorithm):
        def __init__(self,
                    num_sample_training=4,
                    input_shape=(3, 224, 224),
                    bn_training_mode=False,
                    **kwargs):
            super(AutoSlim, self).__init__(**kwargs)
            pass

        def train_step(self, data, optimizer):
            pass
    ```

2. Develop new algorithm components (optional)

    AutoSlim can directly use class `RatioPruner` as core functions provider. If pruners provided in MMRazor don't meet your needs, you can develop new algorithm components for your algorithm like `RatioPruner`. We will take `RatioPruner` as an example to introduce how to develop a new algorithm component:

    a. Create a new file `mmrazor/models/pruners/ratio_pruning.py`, class `RatioPruner` can  inherits from StructurePruner

    b. Finish the functions you need, eg: `sample_subnet`, `set_subnet` and so on

   ```python
   from mmrazor.models.builder import PRUNERS, build_mutable
   from .structure_pruning import StructurePruner

   @PRUNERS.register_module()
   class RatioPruner(StructurePruner):
       def __init__(self, **kwargs):
           super().__init__(**kwargs)

       def sample_subnet(self):
           pass

       def set_subnet(self, subnet_dict):
           pass

       ....

       # supernet is a kind of architecture in `mmrazor/models/architectures/`
       def prepare_from_supernet(self, supernet):
           pass
   ```

   c. Import the module in `mmrazor/models/pruners/__init__.py`

   ```python
    from .ratio_pruning import RatioPruner

    __all__ = [..., 'RatioPruner']
    ```

3. Rewrite its train_step

    Develop key logic of your algorithm in function`train_step`

    ```python
    from mmrazor.models.builder import ALGORITHMS
    from .base import BaseAlgorithm

    @ALGORITHMS.register_module()
    class AutoSlim(BaseAlgorithm):
        def __init__(self,
                    num_sample_training=4,
                    input_shape=(3, 224, 224),
                    bn_training_mode=False,
                    **kwargs):
            super(AutoSlim, self).__init__(**kwargs)
            pass

        def train_step(self, data, optimizer):
            optimizer.zero_grad()
            losses = dict()
            if not self.retraining:
                ...
            else:
                ...
            optimizer.step()
            loss, log_vars = self._parse_losses(losses)
            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
            return outputs
    ```

4. Add your custom functions (optional)

    After finishing your key logic in function train_step, if you also need other custom functions, you can add them in class `AutoSlim`.

5. Import the class

    You can either add the following line to `mmrazor/models/algorithms/__init__.py`

    ```python
    from .autoslim import AutoSlim

    __all__ = [..., 'AutoSlim']
    ```

    Or alternatively add

    ```python
    custom_imports = dict(
        imports=['mmrazor.models.algorithms.spos'],
        allow_failed_imports=False)
    ```

    to the config file to avoid modifying the original code.

6. Use the algorithm in your config file

    ```python
    algorithm = dict(
        type='AutoSlim',
        architecture=...,
        pruner=dict(type='RatioPruner', ...),
        retraining=...)
    ```
