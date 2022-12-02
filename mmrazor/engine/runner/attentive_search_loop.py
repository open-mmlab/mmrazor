# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.registry import LOOPS
from .evolution_search_loop import EvolutionSearchLoop


@LOOPS.register_module()
class AttentiveSearchLoop(EvolutionSearchLoop):
    """Loop for evolution searching with attentive tricks from AttentiveNAS.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        max_epochs (int): Total searching epochs. Defaults to 20.
        max_keep_ckpts (int): The maximum checkpoints of searcher to keep.
            Defaults to 3.
        resume_from (str, optional): Specify the path of saved .pkl file for
            resuming searching.
        num_candidates (int): The length of candidate pool. Defaults to 50.
        top_k (int): Specify top k candidates based on scores. Defaults to 10.
        num_mutation (int): The number of candidates got by mutation.
            Defaults to 25.
        num_crossover (int): The number of candidates got by crossover.
            Defaults to 25.
        mutate_prob (float): The probability of mutation. Defaults to 0.1.
        flops_range (tuple, optional): It is used for screening candidates.
        resource_estimator_cfg (dict): The config for building estimator, which
            is be used to estimate the flops of sampled subnet. Defaults to
            None, which means default config is used.
        score_key (str): Specify one metric in evaluation results to score
            candidates. Defaults to 'accuracy_top-1'.
        init_candidates (str, optional): The candidates file path, which is
            used to init `self.candidates`. Its format is usually in .yaml
            format. Defaults to None.
    """

    def _init_pareto(self):
        # TODO (gaoyang): Fix apis with mmrazor2.0
        for k, v in self.constraints.items():
            if not isinstance(v, (list, tuple)):
                self.constraints[k] = (0, v)

        assert len(self.constraints) == 1, 'Only accept one kind constrain.'
        self.pareto_candidates = dict()
        constraints = list(self.constraints.items())[0]
        discretize_step = self.pareto_mode['discretize_step']
        ds = discretize_step
        # find the left bound
        while ds + 0.5 * discretize_step < constraints[1][0]:
            ds += discretize_step
        self.pareto_candidates[ds] = []
        # find the right bound
        while ds - 0.5 * discretize_step < constraints[1][1]:
            self.pareto_candidates[ds] = []
            ds += discretize_step
