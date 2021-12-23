# Toturial 4: Customize NAS algorithms

Here we show how to develop new NAS algorithms with an example of SPOS.

1. Register a new algorithm

    Create a new file `mmrazor/models/algorithms/spos.py`, class `SPOS` inherits from class `BaseAlgorithm`

    ```Python
    from mmrazor.models.builder import ALGORITHMS
    from .base import BaseAlgorithm

    @ALGORITHMS.register_module()
    class SPOS(BaseAlgorithm):
        def __init__(self, **kwargs):
            super(SPOS, self).__init__(**kwargs)
            pass

        def train_step(self, data, optimizer):
            pass
    ```

2. Develop new algorithm components (optional)

    SPOS can directly use class `OneShotMutator` as core functions provider. If mutators provided in MMRazor don't meet your needs, you can develop new algorithm components for your algorithm like `OneShotMutator`,  we will take `OneShotMutator` as an example to introduce how to develop a new algorithm component:

    a. Create a new file `mmrazor/models/mutators/one_shot_mutator.py`, class `OneShotMutator` inherits from class `BaseMutator`

    b. Finish the functions you need in `OneShotMutator`, eg: `sample_subnet`, `set_subnet` and so on

        ```Python
        from mmrazor.models.builder import MUTATORS
        from .base import BaseMutator


        @MUTATORS.register_module()
        class OneShotMutator(BaseMutator):

            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            @staticmethod
            def get_random_mask(space_info):
                pass

            def sample_subnet(self):
                pass

            def set_subnet(self, subnet_dict):
                pass

            @staticmethod
            def reset_in_subnet(m, in_subnet=True):
                pass

            def set_chosen_subnet(self, subnet_dict):
                pass

            def mutation(self, subnet_dict, prob=0.1):
                pass

            @staticmethod
            def crossover(subnet_dict1, subnet_dict2):
                pass
        ```

    c. Import the new mutator

        You can either add the following line to `mmrazor/models/mutators/__init__.py`

        ```CoffeeScript
        from .one_shot_mutator import OneShotMutator
        ```

        or alternatively add

        ```Python
        custom_imports = dict(
            imports=['mmrazor.models.mutators.one_shot_mutator'],
            allow_failed_imports=False)
        ```

        to the config file to avoid modifying the original code.

    d. Use the algorithm component in your config file

        ```Python
        mutator = dict(
            type='OneShotMutator',
            ...)
        ```

3.  Rewrite its train_step

    Develop key logic of your algorithm in function`train_step`

    ```Python
    @ALGORITHMS.register_module()
    class SPOS(BaseAlgorithm):
        def __init__(self, **kwargs):
            super(SPOS, self).__init__(**kwargs)
            pass

        def train_step(self, data, optimizer):
            if self.retraining:
                outputs = super(SPOS, self).train_step(data, optimizer)
            else:
                subnet_dict = self.mutator.sample_subnet()
                self.mutator.set_subnet(subnet_dict)
                outputs = super(SPOS, self).train_step(data, optimizer)
            return outputs
    ```

4. Add your custom functions (optional)

    After finishing your key logic in function `train_step`, if you also need other custom functions, you can add them in class `SPOS` as follows.

    ```Python
    @ALGORITHMS.register_module()
    class SPOS(BaseAlgorithm):
        def __init__(self, **kwargs):
            super(SPOS, self).__init__(**kwargs)
            pass

        def _init_flops(self):
            pass

        def get_subnet_flops(self):
            pass

        def train_step(self, data, optimizer):
            if self.retraining:
                outputs = super(SPOS, self).train_step(data, optimizer)
            else:
                subnet_dict = self.mutator.sample_subnet()
                self.mutator.set_subnet(subnet_dict)
                outputs = super(SPOS, self).train_step(data, optimizer)
            return outputs
    ```

5. Import the class

    You can either add the following line to `mmrazor/models/algorithms/__init__.py`

    ```CoffeeScript
    from .spos import SPOS
    ```

    or alternatively add

    ```Python
    custom_imports = dict(
        imports=['mmrazor.models.algorithms.spos'],
        allow_failed_imports=False)
    ```

    to the config file to avoid modifying the original code.

6. Use the algorithm in your config file

    ```Python
    algorithm = dict(
        type='SPOS',
        mutator=mutator,   # you can use it here if you developed new algorithm components
        ...
    )
    ```
