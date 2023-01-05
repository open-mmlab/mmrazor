# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

from mmengine.evaluator import Evaluator
from mmengine.hooks import CheckpointHook
from mmengine.runner import ValLoop, BaseLoop, TestLoop, EpochBasedTrainLoop
from torch.utils.data import DataLoader

from mmrazor.models.utils import add_prefix
from mmrazor.registry import LOOPS, TASK_UTILS

import mmrazor.models.architectures.backbones.global_utils as global_utils
import torch, time, random
import numpy as np

from mmrazor.models.architectures.backbones import MasterNet
from mmrazor.models.architectures.backbones.PlainNet.basic_blocks import build_netblock_list_from_str
# from mmrazor.models.architectures.backbones.ZeroShotProxy import compute_zen_score, compute_te_nas_score, compute_syncflow_score, compute_gradnorm_score, compute_NASWOT_score
from mmrazor.models.architectures.backbones.ZeroShotProxy import compute_zen_score

def get_splitted_structure_str(AnyPlainNet, structure_str, num_classes):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert hasattr(the_net, 'split')
    splitted_net_str = the_net.split(split_layer_threshold=6)
    return splitted_net_str

def get_new_random_structure_str(AnyPlainNet, structure_str, num_classes, get_search_space_func,
                                 num_replaces=1):
    the_net = AnyPlainNet(num_classes=num_classes, plainnet_struct=structure_str, no_create=True)
    assert isinstance(the_net, MasterNet)
    selected_random_id_set = set()
    for replace_count in range(num_replaces):
        random_id = random.randint(0, len(the_net.block_list) - 1)
        if random_id in selected_random_id_set:
            continue
        selected_random_id_set.add(random_id)
        to_search_student_blocks_list_list = get_search_space_func(the_net.block_list, random_id)

        to_search_student_blocks_list = [x for sublist in to_search_student_blocks_list_list for x in sublist]
        new_student_block_str = random.choice(to_search_student_blocks_list)

        if len(new_student_block_str) > 0:
            # new_student_block = PlainNet.create_netblock_list_from_str(new_student_block_str, no_create=True)
            # new_student_block = create_netblock_list_from_str(new_student_block_str, no_create=True)
            new_student_block = build_netblock_list_from_str(new_student_block_str, no_create=True)
            assert len(new_student_block) == 1
            new_student_block = new_student_block[0]
            if random_id > 0:
                last_block_out_channels = the_net.block_list[random_id - 1].out_channels
                new_student_block.set_in_channels(last_block_out_channels)
            the_net.block_list[random_id] = new_student_block
        else:
            # replace with empty block
            the_net.block_list[random_id] = None
    pass  # end for

    # adjust channels and remove empty layer
    tmp_new_block_list = [x for x in the_net.block_list if x is not None]
    last_channels = the_net.block_list[0].out_channels
    for block in tmp_new_block_list[1:]:
        block.set_in_channels(last_channels)
        last_channels = block.out_channels
    the_net.block_list = tmp_new_block_list

    new_random_structure_str = the_net.split(split_layer_threshold=6)
    return new_random_structure_str


@LOOPS.register_module()
class ZeroShotLoop(EpochBasedTrainLoop):
    """Loop for subnet validation in NAS with BN re-calibration.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
        evaluate_fixed_subnet (bool): Whether to evaluate a fixed subnet only
            or not. Defaults to False.
        calibrate_sample_num (int): The number of images to compute the true
            average of per-batch mean/variance instead of the running average.
            Defaults to 4096.
        estimator_cfg (dict, Optional): Used for building a resource estimator.
            Defaults to dict(type='mmrazor.ResourceEstimator').
    """

    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        evaluator: Union[Evaluator, Dict, List],
        search_space: Optional[str] = '',
        plainnet_struct_txt: Optional[str] = '',
        zero_shot_score: Optional[str] = 'Zen',
        max_epochs: int = 20,
        population_size: int = 512,
        num_classes: int = 1000,
        max_layers: int = 10,
        budget_model_size: Optional[int] = None,
        budget_flops: Optional[int] = None,
        batch_size: Optional[int] = 64,
        input_image_size: Optional[int] = 224,
        gamma: Optional[float] = 1e-2,
        fp16: bool = False,
        evaluate_fixed_subnet: bool = False,
        calibrate_sample_num: int = 4096,
        estimator_cfg: Optional[Dict] = dict(type='mmrazor.ResourceEstimator')
    ) -> None:
        super().__init__(runner, dataloader, max_epochs)

        # if self.runner.distributed:
        #     model = self.runner.model.module
        # else:
        model = self.runner.model

        self.gpu=0
        self.gamma = gamma
        self.input_image_size = input_image_size
        self.batch_size = batch_size

        self.zero_shot_score = zero_shot_score
        self.population_size = population_size
        self.num_classes = num_classes
        self.max_layers = max_layers
        self.budget_model_size = budget_model_size
        self.budget_flops = budget_flops
        self.model = model
        self.evaluate_fixed_subnet = evaluate_fixed_subnet
        self.calibrate_sample_num = calibrate_sample_num
        # self.estimator = TASK_UTILS.build(estimator_cfg)

        # remove CheckpointHook to avoid extra problems.
        # for hook in self.runner._hooks:
        #     if isinstance(hook, CheckpointHook):
        #         self.runner._hooks.remove(hook)
        #         break
        
        ## 
        # load search space config .py file
        self.select_search_space = global_utils.load_py_module_from_path(search_space)

        # load masternet
        # fix_subnet = plainnet_struct_txt
        self.initial_structure_str = str(self.model.architecture.backbone)

    def run(self) -> None:
        """Launch searching."""
        self.runner.call_hook('before_train')

        # if self.predictor_cfg is not None:
        #     self._init_predictor()

        # if self.resume_from:
        #     self._resume()

        self.popu_structure_list = []
        self.popu_zero_shot_score_list = []
        self.popu_latency_list = []

        # while self._epoch < self._max_epochs:
        self.start_timer = time.time()
        for loop_count in range(self._max_epochs):
            self.run_epoch(loop_count)
            # self._save_searcher_ckpt()

        # self._save_best_fix_subnet()

        self.runner.call_hook('after_train')

    def run_epoch(self, loop_count) -> None:
        """Iterate one epoch.

        Steps:
            1. Sample some new candidates from the supernet. Then Append them
                to the candidates, Thus make its number equal to the specified
                number.
            2. Validate these candidates(step 1) and update their scores.
            3. Pick the top k candidates based on the scores(step 2), which
                will be used in mutation and crossover.
            4. Implement Mutation and crossover, generate better candidates.
        """

        while len(self.popu_structure_list) > self.population_size: # 512个候选网络
            min_zero_shot_score = min(self.popu_zero_shot_score_list)
            tmp_idx = self.popu_zero_shot_score_list.index(min_zero_shot_score)
            self.popu_zero_shot_score_list.pop(tmp_idx)
            self.popu_structure_list.pop(tmp_idx)
            self.popu_latency_list.pop(tmp_idx)
        pass

        if loop_count >= 1 and loop_count % 100 == 0:
            max_score = max(self.popu_zero_shot_score_list)
            min_score = min(self.popu_zero_shot_score_list)
            elasp_time = time.time() - self.start_timer
            self.runner.logger.info(f'loop_count={loop_count}/{self._max_epochs}, max_score={max_score:4g}, min_score={min_score:4g}, time={elasp_time/3600:4g}h')

        # ----- generate a random structure ----- #
        random_structure_str = self.sample_candidates()

        the_model = None
        # 经过筛选 
        # max_layers / budget_model_size / budget_flops / budget_latency=0.0001
        if self.max_layers is not None: # 10
            if the_model is None:
                the_model = MasterNet(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False) # 这里去除VCNN指的是只有包含RELU激活函数的卷积层构成的网络，不包含BN、残差块等算子，并且只取到GAP(global average pool)层的前一层以保留更多的信息。
            the_layers = the_model.get_num_layers()
            if self.max_layers < the_layers:
                return

        if self.budget_model_size is not None:
            if the_model is None:
                the_model = MasterNet(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_size = the_model.get_model_size()
            if self.budget_model_size < the_model_size:
                return

        if self.budget_flops is not None:
            if the_model is None:
                the_model = MasterNet(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                        no_create=True, no_reslink=False)
            the_model_flops = the_model.get_FLOPs(self.input_image_size)
            if self.budget_flops < the_model_flops:
                return

        the_latency = np.inf
        """ # latency 
        if self.budget_latency is not None: # 0.0001
            the_latency = get_latency(MasterNet, random_structure_str, gpu, args)
            if self.budget_latency < the_latency:
                return
        """

        the_nas_core = self.compute_nas_score(MasterNet, random_structure_str, self.gpu)

        self.popu_structure_list.append(random_structure_str)
        self.popu_zero_shot_score_list.append(the_nas_core)
        self.popu_latency_list.append(the_latency)

        self._epoch += 1

    def sample_candidates(self):
        # ----- generate a random structure ----- #
        if len(self.popu_structure_list) <= 10:
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=MasterNet, structure_str=self.initial_structure_str, num_classes=self.num_classes,
                get_search_space_func=self.select_search_space.gen_search_space, num_replaces=1)
        else:
            tmp_idx = random.randint(0, len(self.popu_structure_list) - 1)
            tmp_random_structure_str = self.popu_structure_list[tmp_idx]
            random_structure_str = get_new_random_structure_str(
                AnyPlainNet=MasterNet, structure_str=tmp_random_structure_str, num_classes=self.num_classes,
                get_search_space_func=self.select_search_space.gen_search_space, num_replaces=2)

        random_structure_str = get_splitted_structure_str(MasterNet, random_structure_str,
                                                          num_classes=self.num_classes)
        return random_structure_str

    def compute_nas_score(self, AnyPlainNet, random_structure_str, gpu):
        # compute network zero-shot proxy score
        the_model = AnyPlainNet(num_classes=self.num_classes, plainnet_struct=random_structure_str,
                                no_create=False, no_reslink=True)
        the_model = the_model.cuda(gpu)
        try:
            if self.zero_shot_score == 'Zen':
                the_nas_core_info = compute_zen_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                        resolution=self.input_image_size,
                                                                        mixup_gamma=self.gamma, batch_size=self.batch_size,
                                                                        repeat=1)
                the_nas_core = the_nas_core_info['avg_nas_score']
            """
            elif self.zero_shot_score == 'TE-NAS':
                the_nas_core = compute_te_nas_score.compute_NTK_score(model=the_model, gpu=gpu,
                                                                    resolution=self.input_image_size,
                                                                    batch_size=self.batch_size)

            elif self.zero_shot_score == 'Syncflow':
                the_nas_core = compute_syncflow_score.do_compute_nas_score(model=the_model, gpu=gpu,
                                                                        resolution=self.input_image_size,
                                                                        batch_size=self.batch_size)

            elif self.zero_shot_score == 'GradNorm':
                the_nas_core = compute_gradnorm_score.compute_nas_score(model=the_model, gpu=gpu,
                                                                        resolution=self.input_image_size,
                                                                        batch_size=self.batch_size)

            elif self.zero_shot_score == 'Flops':
                the_nas_core = the_model.get_FLOPs(self.input_image_size)

            elif self.zero_shot_score == 'Params':
                the_nas_core = the_model.get_model_size()

            elif self.zero_shot_score == 'Random':
                the_nas_core = np.random.randn()

            elif self.zero_shot_score == 'NASWOT':
                the_nas_core = compute_NASWOT_score.compute_nas_score(gpu=gpu, model=the_model,
                                                                    resolution=self.input_image_size,
                                                                    batch_size=self.batch_size)
            """
        except Exception as err:
            # logging.info(str(err))
            # logging.info('--- Failed structure: ')
            # logging.info(str(the_model))
            # raise err
            the_nas_core = -9999


        del the_model
        torch.cuda.empty_cache()
        return the_nas_core
