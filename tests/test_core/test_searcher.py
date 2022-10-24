# Copyright (c) OpenMMLab. All rights reserved.
import copy
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader, Dataset

from mmrazor.core.builder import SEARCHERS


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


class TestEvolutionSearcher(TestCase):

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        self.searcher_cfg = dict(
            type='EvolutionSearcher',
            metrics='bbox',
            score_key='bbox_mAP',
            constraints=dict(flops=300 * 1e6),
            candidate_pool_size=50,
            candidate_top_k=10,
            max_epoch=20,
            num_mutation=20,
            num_crossover=20,
        )
        self.dataloader = DataLoader(ToyDataset(), collate_fn=collate_fn)
        self.test_fn = MagicMock()
        self.logger = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.work_dir)

    @patch('mmrazor.models.algorithms.DetNAS')
    def test_init(self, mock_algo):
        cfg = copy.deepcopy(self.searcher_cfg)
        cfg['algorithm'] = mock_algo
        cfg['dataloader'] = self.dataloader
        cfg['test_fn'] = self.test_fn
        cfg['work_dir'] = self.work_dir
        cfg['logger'] = self.logger
        searcher = SEARCHERS.build(cfg)
        assert hasattr(searcher, 'algorithm')
        assert hasattr(searcher, 'logger')

        cfg['num_mutation'] = 40
        with self.assertRaises(ValueError):
            searcher = SEARCHERS.build(cfg)
