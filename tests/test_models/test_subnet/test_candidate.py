# Copyright (c) OpenMMLab. All rights reserved.

from collections import UserList
from unittest import TestCase

from mmrazor.structures import Candidates


class TestCandidates(TestCase):

    def setUp(self) -> None:
        self.fake_subnet = {'1': 'choice1', '2': 'choice2'}
        self.fake_subnet_with_resource = {
            str(self.fake_subnet): {
                'score': 0.,
                'flops': 50.,
                'params': 0.,
                'latency': 0.
            }
        }
        self.fake_subnet_with_score = {
            str(self.fake_subnet): {
                'score': 99.,
                'flops': 0.,
                'params': 0.,
                'latency': 0.
            }
        }
        self.has_flops_network = {
            str(self.fake_subnet): {
                'flops': 50.,
            }
        }

    def test_init(self):
        # initlist is None
        candidates = Candidates()
        self.assertEqual(len(candidates.data), 0)
        # initlist is list
        data = [self.fake_subnet] * 2
        candidates = Candidates(data)
        self.assertEqual(len(candidates.data), 2)
        # initlist is UserList
        data = UserList([self.fake_subnet] * 2)
        self.assertEqual(len(candidates.data), 2)
        self.assertEqual(candidates.resources('flops'), [-1, -1])
        # initlist is list(Dict[str, Dict])
        candidates = Candidates([self.has_flops_network] * 2)
        self.assertEqual(candidates.resources('flops'), [50., 50.])

    def test_scores(self):
        # test property: scores
        data = [self.fake_subnet_with_score] * 2
        candidates = Candidates(data)
        self.assertEqual(candidates.scores, [99., 99.])

    def test_resources(self):
        data = [self.fake_subnet_with_resource] * 2
        candidates = Candidates(data)
        self.assertEqual(candidates.resources('flops'), [50., 50.])

    def test_subnets(self):
        # test property: subnets
        data = [self.fake_subnet] * 2
        candidates = Candidates(data)
        self.assertEqual(candidates.subnets, [self.fake_subnet] * 2)

    def test_append(self):
        # item is dict
        candidates = Candidates()
        candidates.append(self.fake_subnet)
        self.assertEqual(len(candidates), 1)
        # item is List
        candidates = Candidates()
        candidates.append([self.fake_subnet_with_score])
        # item is Candidates
        candidates_2 = Candidates([self.fake_subnet_with_resource])
        candidates.append(candidates_2)
        self.assertEqual(len(candidates), 2)

    def test_insert(self):
        # item is dict
        candidates = Candidates(self.fake_subnet_with_score)
        candidates.insert(1, self.fake_subnet)
        self.assertEqual(len(candidates), 2)
        # item is List
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.insert(1, self.fake_subnet_with_score)
        self.assertEqual(len(candidates), 2)

    def test_extend(self):
        # other is list
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.extend([self.fake_subnet])
        self.assertEqual(len(candidates), 2)
        # other is Candidates
        candidates = Candidates([self.fake_subnet_with_score])
        candidates_2 = Candidates([self.fake_subnet_with_resource])
        candidates.extend(candidates_2)
        self.assertEqual(len(candidates), 2)

    def test_set_resource(self):
        # test set_resource
        candidates = Candidates([self.fake_subnet])
        for kk in ['flops', 'params', 'latency']:
            self.assertEqual(candidates.resources(kk)[0], -1)
            candidates.set_resource(0, 49.9, kk)
            self.assertEqual(candidates.resources(kk)[0], 49.9)
        candidates.insert(0, self.fake_subnet_with_resource)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates.resources('flops'), [50., 49.9])
        self.assertEqual(candidates.resources('latency'), [0., 49.9])
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.set_resource(0, 100.0, 'score')
        self.assertEqual(candidates.scores[0], 100.)
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.set_resource(0, 100.0, 'score')
        candidates.extend(UserList([self.fake_subnet_with_resource]))
        candidates.set_resource(1, 99.9, 'score')
        self.assertEqual(candidates.scores, [100., 99.9])

    def test_update_resources(self):
        # test update_resources
        candidates = Candidates([self.fake_subnet])
        candidates.append([self.fake_subnet_with_score])
        candidates_2 = Candidates(self.fake_subnet_with_resource)
        candidates.append(candidates_2)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates.resources('flops'), [-1, 0., 50.])
        self.assertEqual(candidates.resources('latency'), [-1, 0., 0.])
        resources = [{'flops': -2}, {'latency': 4.}]
        candidates.update_resources(resources, start=1)
        self.assertEqual(candidates.resources('flops'), [-1, -2, 50.])
        self.assertEqual(candidates.resources('latency'), [-1, 0., 4])
        candidates.update_resources(resources, start=0)
        self.assertEqual(candidates.resources('flops'), [-2, -2, 50.])
        self.assertEqual(candidates.resources('latency'), [-1, 4., 4.])

    def test_sort(self):
        # test set_sort
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.extend(UserList([self.fake_subnet_with_resource]))
        candidates.insert(0, self.fake_subnet)
        candidates.set_resource(0, 100., 'score')
        candidates.set_resource(2, 98., 'score')
        self.assertEqual(candidates.scores, [100., 99., 98.])
        candidates.sort_by(key_indicator='score', reverse=False)
        self.assertEqual(candidates.scores, [98., 99., 100.])
        candidates.sort_by(key_indicator='latency')
        self.assertEqual(candidates.scores, [98., 99., 100.])
        candidates.sort_by(key_indicator='flops', reverse=False)
        self.assertEqual(candidates.scores, [100., 99., 98.])
