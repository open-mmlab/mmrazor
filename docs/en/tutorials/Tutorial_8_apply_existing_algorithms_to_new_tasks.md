# Tutorial 8: Apply existing algorithms to new tasks

Here we show how to apply existing algorithms to other existing tasks with an example of SPOS & DetNAS.

1. Register a new algorithm for the other existing task with an existing algorithm

    Create a new file `mmrazor/models/algorithms/detnas.py`, class `DetNAS` inherits from class `SPOS`

    ```python
    from mmrazor.models.builder import ALGORITHMS
    from .spos import SPOS

    @ALGORITHMS.register_module()
    class DetNAS(SPOS):

        def __init__(self, **kwargs):
            super(DetNAS, self).__init__(**kwargs)
    ```

2. Add your custom functions (optional)

    If you need other custom functions according to the other existing task, you can add them in class `DetNAS` as follows.

    ```python
    class DetNAS(SPOS):

        ...

        def _init_flops(self):
            flops_model = copy.deepcopy(self.architecture)
            flops_model = revert_sync_batchnorm(flops_model)
            flops_model.eval()
            flops, params = get_model_complexity_info(flops_model.model.backbone,
                                                    self.input_shape)
            flops_lookup = dict()
            for name, module in flops_model.named_modules():
                flops = getattr(module, '__flops__', 0)
                flops_lookup[name] = flops
            del (flops_model)

            for name, module in self.architecture.named_modules():
                module.__flops__ = flops_lookup[name]

        ...

    ```

3. Import the class

    You can either add the following line to `mmrazor/models/algorithms/__init__.py`

    ```python
    from .detnas import DetNAS
    ```

    or alternatively add

    ```python
    custom_imports = dict(
        imports=['mmrazor.models.algorithms.detnas'],
        allow_failed_imports=False)
    ```

    to the config file to avoid modifying the original code.

4. Use the algorithm in your config file

    ```python
    algorithm = dict(
        type='DetNAS',
        ...
    )
    ```
