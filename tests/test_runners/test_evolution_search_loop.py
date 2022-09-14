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
        return nn.Conv2d

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
            flops_range=None,
            score_key='coco/bbox_mAP')
        self.train_cfg = Config(train_cfg)
        self.runner = MagicMock(spec=ToyRunner)
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
        fake_candidates = Candidates((fake_subnet, 0.))
        init_candidates_path = os.path.join(self.temp_dir, 'candidates.yaml')
        fileio.dump(fake_candidates, init_candidates_path)
        loop_cfg.init_candidates = init_candidates_path
        loop = LOOPS.build(loop_cfg)
        self.assertIsInstance(loop, EvolutionSearchLoop)
        self.assertEqual(loop.candidates, fake_candidates)

    @patch('mmrazor.engine.runner.evolution_search_loop.export_fix_subnet')
    @patch(
        'mmrazor.engine.runner.evolution_search_loop.get_model_complexity_info'
    )
    def test_run_epoch(self, mock_flops, mock_export_fix_subnet):
        # test_run_epoch: distributed == False
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop._epoch = 1
        self.runner.distributed = False
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        self.runner.model.sample_subnet = MagicMock(return_value=fake_subnet)
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 2)

        # test_run_epoch: distributed == True
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop._epoch = 1
        self.runner.distributed = True
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        self.runner.model.sample_subnet = MagicMock(return_value=fake_subnet)
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 2)

        # test_check_constraints
        loop_cfg.flops_range = (0, 100)
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop._epoch = 1
        self.runner.distributed = True
        self.runner.work_dir = self.temp_dir
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        loop.model.sample_subnet = MagicMock(return_value=fake_subnet)
        mock_flops.return_value = (50., 1)
        mock_export_fix_subnet.return_value = fake_subnet
        loop.run_epoch()
        self.assertEqual(len(loop.candidates), 4)
        self.assertEqual(len(loop.top_k_candidates), 2)
        self.assertEqual(loop._epoch, 2)

    @patch('mmrazor.engine.runner.evolution_search_loop.export_fix_subnet')
    def test_run(self, mock_export_fix_subnet):
        # test a new search: resume == None
        loop_cfg = copy.deepcopy(self.train_cfg)
        loop_cfg.runner = self.runner
        loop_cfg.dataloader = self.dataloader
        loop_cfg.evaluator = self.evaluator
        loop = LOOPS.build(loop_cfg)
        self.runner.rank = 0
        loop._epoch = 1
        fake_subnet = {'1': 'choice1', '2': 'choice2'}
        self.runner.work_dir = self.temp_dir
        loop.update_candidate_pool = MagicMock()
        loop.val_candidate_pool = MagicMock()
        loop.gen_mutation_candidates = \
            MagicMock(return_value=[fake_subnet]*loop.num_mutation)
        loop.gen_crossover_candidates = \
            MagicMock(return_value=[fake_subnet]*loop.num_crossover)
        loop.top_k_candidates = Candidates([(fake_subnet, 1.0),
                                            (fake_subnet, 0.9)])
        mock_export_fix_subnet.return_value = fake_subnet
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
