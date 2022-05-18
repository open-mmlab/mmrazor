# Tutorial 7: Customize mixed algorithms with our algorithm components

Here we show how to customize mixed algorithms with our algorithm components. We take the slimmable training in autoslim as an example.

The sandwich rule and inplace distillation were introduced to enhance the training process. The sandwich rule means that we train the model at smallest width, largest width and (n − 2) random widths, instead of n random widths. By inplace distillation, we use the predicted label of the model at the largest width as the training label for other widths, while for the largest width we use ground truth. So both the KD algorithm and the pruning algorithm are used in slimmable training.

1. In the distillation part, we can directly use SelfDistiller in `mmrazor/models/distillers/self_distiller.py`. If distillers provided in MMRazor don't meet your needs, you can develop new algorithm components for your algorithm as step2 in Tutorial 6.

2. As the slimmable training is the first step of `Autoslim`, we do not need to register a new algorithm, but rewrite the `train_step`function in AutoSlim as follows:

   ```python
   from mmrazor.models.builder import ALGORITHMS
   from .base import BaseAlgorithm

   @ALGORITHMS.register_module()
   class AutoSlim(BaseAlgorithm):
       def train_step(self, data, optimizer):
           optimizer.zero_grad()
           losses = dict()
           if not self.retraining:
               #
           else:
               ...
           optimizer.step()
           loss, log_vars = self._parse_losses(losses)
           outputs = dict(
               loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
           return outputs
   ```

3. Use the algorithm in your config file

   ```python
   algorithm = dict(
       type='AutoSlim',
       architecture=...,
       pruner=dict(
           type='RatioPruner',
           ratios=(2 / 12, 3 / 12, 4 / 12, 5 / 12, 6 / 12, 7 / 12, 8 / 12, 9 / 12,
                   10 / 12, 11 / 12, 1.0)),
       distiller=dict(
           type='SelfDistiller',
           components=...),
       retraining=False)
   ```
