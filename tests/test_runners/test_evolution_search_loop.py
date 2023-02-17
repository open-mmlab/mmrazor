# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from mmengine import fileio
from mmengine.config import Config
from torch.utils.data import DataLoader, Dataset

from mmrazor.engine import EvolutionSearchLoop
from mmrazor.models import OneShotMutableOP
from mmrazor.registry import LOOPS
from mmrazor.structures import Candidates


def collate_fn(data_batch):
    return data_batch


class ToyDataset(Dataset):
    METAINFO = dict()  # type: ignore
    data = torch.randn(12, 2)
    label = torch.ones(12)

    @property
    def metainfo(self):
        return self.METAINFO

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        return dict(inputs=self.data[index], data_sample=self.label[index])


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.architecture = nn.Conv2d(1, 1, 1)

    def forward(self, x):
        return self.architecture(x)


class ToyRunner:

    @property
    def distributed(self):
        pass

    @property
    def rank(self):
        pass

    @property
    def epoch(self):
        pass

    @property
    def work_dir(self):
        pass

    def model(self):
        return ToyModel()

    def logger(self):
        pass

    def call_hook(self, fn_name: str):
        pass

    def visualizer(self):
        pass


class TestEvolutionSearchLoop(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        train_cfg = dict(
            type='EvolutionSearchLoop',
            max_epochs=4,
            max_keep_ckpts=3,
            resume_from=None,
            num_candidates=4,
            top_k=2,
            num_mutation=2,
            num_crossover=2,
            mutate_prob=0.1,
            constraints_range=dict(flops=(0, 330)),
            score_key='coco/bbox_mAP')
        self.train_cfg = Config(train_cfg)
        self.runner = MagicMock(spec=ToyRunner)
        self.runner.train_dataloader = MagicMock()
        self.dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        self.evaluator = MagicMock()
        self.calibrate_bn_statistics = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # test_init: dataloader and evaluator are instances
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.assertIsInstance(loop, EvolutionSearchLoop)

        # test init_candidates is not None
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        fake_candidates = Candidates(fake_subnet)
        init_candidates_path = os.path.join(self.temp_dir, 'candidates.yaml')
        fileio.dump(fake_candidates, init_candidates_path)
        loop_cfg.init_candidates = init_candidates_path
        loop = LOOPS.build(loop_cfg)
        self.assertIsInstance(loop, EvolutionSearchLoop)
        self.assertEqual(loop.candidates, fake_candidates)

    @patch('mmrazor.structures.subnet.fix_subnet.load_fix_subnet')
    @patch('mmrazor.structures.subnet.fix_subnet.export_fix_subnet')
    @patch('mmrazor.models.task_modules.estimators.resource_estimator.'
           'get_model_flops_params')
    def test_run_epoch(self, flops_params, mock_export_fix_subnet,
                       load_status):
        # test_run_epoch: distributed == False
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        self.runner.distributed = False
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        loop.model.mutator.sample_choices = MagicMock(return_value=fake_subnet)
        mock_export_fix_subnet.return_value = (fake_subnet, self.runner.model)
        load_status.return_value = True
        flops_params.return_value = 0, 0
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 1)

        # test_run_epoch: distributed == True
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        self.runner.distributed = True
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        self.runner.model.mutator.sample_choices = MagicMock(
            return_value=fake_subnet)
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 1)

        # test_check_constraints
        loop_cfg.constraints_range = dict(params=(0, 100))
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        self.runner.distributed = True
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        loop.model.mutator.sample_choices = MagicMock(return_value=fake_subnet)
        flops_params.return_value = (50., 1)
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 1)

    @patch('mmrazor.structures.subnet.fix_subnet.export_fix_subnet')
    @patch('mmrazor.models.task_modules.estimators.resource_estimator.'
           'get_model_flops_params')
    def test_run_loop(self, mock_flops, mock_export_fix_subnet):
        # test a new search: resume == None
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop._epoch = 1

        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        mock_export_fix_subnet.return_value = (fake_subnet, self.runner.model)
        self.runner.work_dir = self.temp_dir
        loop.update_candidate_pool = MagicMock()
        loop.val_candidate_pool = MagicMock()

        mutation_candidates = Candidates([fake_subnet] * loop.num_mutation)
        for i in range(loop.num_mutation):
            mutation_candidates.set_resource(i, 0.1 + 0.1 * i, 'flops')
            mutation_candidates.set_resource(i, 99 + i, 'score')
        crossover_candidates = Candidates([fake_subnet] * loop.num_crossover)
        for i in range(loop.num_crossover):
            crossover_candidates.set_resource(i, 0.1 + 0.1 * i, 'flops')
            crossover_candidates.set_resource(i, 99 + i, 'score')
        loop.gen_mutation_candidates = \
            MagicMock(return_value=mutation_candidates)
        loop.gen_crossover_candidates = \
            MagicMock(return_value=crossover_candidates)
        loop.candidates = Candidates([fake_subnet] * 4)
        mock_flops.return_value = (0.5, 101)
        torch.save = MagicMock()
        loop.run()
        assert os.path.exists(
            os.path.join(self.temp_dir, 'best_fix_subnet.yaml'))
        self.assertEqual(loop._epoch, loop._max_epochs)
        assert os.path.exists(
            os.path.join(self.temp_dir,
                         f'search_epoch_{loop._max_epochs-1}.pkl'))
        # test resuming search
        loop_cfg.resume_from = os.path.join(
            self.temp_dir, f'search_epoch_{loop._max_epochs-1}.pkl')
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop.run()
        self.assertEqual(loop._max_epochs, 1)


class TestEvolutionSearchLoopWithPredictor(TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        convs = nn.ModuleDict({
            'conv1': nn.Conv2d(3, 8, 1),
            'conv2': nn.Conv2d(3, 8, 1),
            'conv3': nn.Conv2d(3, 8, 1),
        })
        MutableOP = OneShotMutableOP(convs)
        self.search_groups = {0: [MutableOP], 1: [MutableOP]}
        train_cfg = dict(
            type='EvolutionSearchLoop',
            max_epochs=4,
            max_keep_ckpts=3,
            resume_from=None,
            num_candidates=4,
            top_k=2,
            num_mutation=2,
            num_crossover=2,
            mutate_prob=0.1,
            constraints_range=dict(flops=(0, 330)),
            score_key='bbox_mAP',
            predictor_cfg=dict(
                type='MetricPredictor',
                handler_cfg=dict(type='GaussProcessHandler'),
                search_groups=self.search_groups,
                train_samples=4,
            ))
        self.train_cfg = Config(train_cfg)
        self.runner = MagicMock(spec=ToyRunner)
        self.runner.train_dataloader = MagicMock()
        self.dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        self.evaluator = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_init(self):
        # test_init: dataloader and evaluator are instances
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.assertIsInstance(loop, EvolutionSearchLoop)

        # test init_candidates is not None
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        fake_candidates = Candidates(fake_subnet)
        init_candidates_path = os.path.join(self.temp_dir, 'candidates.yaml')
        fileio.dump(fake_candidates, init_candidates_path)
        loop_cfg.init_candidates = init_candidates_path
        loop = LOOPS.build(loop_cfg)
        self.assertIsInstance(loop, EvolutionSearchLoop)
        self.assertEqual(loop.candidates, fake_candidates)

    @patch('mmrazor.structures.subnet.fix_subnet.load_fix_subnet')
    @patch('mmrazor.structures.subnet.fix_subnet.export_fix_subnet')
    @patch('mmrazor.models.task_modules.estimators.resource_estimator.'
           'get_model_flops_params')
    def test_run_epoch(self, flops_params, mock_export_fix_subnet,
                       load_status):
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        self.runner.distributed = False
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        loop.model.mutator.sample_choices = MagicMock(return_value=fake_subnet)
        mock_export_fix_subnet.return_value = (fake_subnet, self.runner.model)
        load_status.return_value = True
        flops_params.return_value = 0, 0
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 1)

        # test_run_epoch: distributed == True
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        self.runner.distributed = True
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        self.runner.model.mutator.sample_choices = MagicMock(
            return_value=fake_subnet)
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 1)

        # test_check_constraints
        loop_cfg.constraints_range = dict(params=(0, 100))
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        self.runner.distributed = True
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        loop.model.mutator.sample_choices = MagicMock(return_value=fake_subnet)
        flops_params.return_value = (50., 1)
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 1)

    @patch('mmrazor.structures.subnet.fix_subnet.export_fix_subnet')
    @patch('mmrazor.models.task_modules.predictor.metric_predictor.'
           'MetricPredictor.model2vector')
    @patch('mmrazor.models.task_modules.estimators.resource_estimator.'
           'get_model_flops_params')
    def test_run_loop(self, mock_flops, mock_model2vector,
                      mock_export_fix_subnet):
        # test a new search: resume == None
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop._epoch = 1

        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        loop.model.mutator.sample_choices = MagicMock(return_value=fake_subnet)
        mock_export_fix_subnet.return_value = (fake_subnet, self.runner.model)

        self.runner.work_dir = self.temp_dir
        loop.update_candidate_pool = MagicMock()
        loop.val_candidate_pool = MagicMock()

        mutation_candidates = Candidates([fake_subnet] * loop.num_mutation)
        for i in range(loop.num_mutation):
            mutation_candidates.set_resource(i, 0.1 + 0.1 * i, 'flops')
            mutation_candidates.set_resource(i, 99 + i, 'score')
        crossover_candidates = Candidates([fake_subnet] * loop.num_crossover)
        for i in range(loop.num_crossover):
            crossover_candidates.set_resource(i, 0.1 + 0.1 * i, 'flops')
            crossover_candidates.set_resource(i, 99 + i, 'score')
        loop.gen_mutation_candidates = \
            MagicMock(return_value=mutation_candidates)
        loop.gen_crossover_candidates = \
            MagicMock(return_value=crossover_candidates)
        loop.candidates = Candidates([fake_subnet] * 4)

        mock_flops.return_value = (0.5, 101)
        mock_model2vector.return_value = dict(
            normal_vector=[0, 1], onehot_vector=[0, 1, 0, 1])
        torch.save = MagicMock()
        loop.run()
        assert os.path.exists(
            os.path.join(self.temp_dir, 'best_fix_subnet.yaml'))
        self.assertEqual(loop._epoch, loop._max_epochs)
        assert os.path.exists(
            os.path.join(self.temp_dir,
                         f'search_epoch_{loop._max_epochs-1}.pkl'))
        # test resuming search
        loop_cfg.resume_from = os.path.join(
            self.temp_dir, f'search_epoch_{loop._max_epochs-1}.pkl')
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop.run()
        self.assertEqual(loop._max_epochs, 1)
