# Copyright (c) OpenMMLab. All rights reserved.

from collections import UserList
from unittest import TestCase

from mmrazor.structures import Candidates


class TestCandidates(TestCase):

    def setUp(self) -> None:
        self.fake_subnet = {'1': 'choice1', '2': 'choice2'}
        self.fake_subnet_with_resource = (self.fake_subnet, 50)
        self.fake_subnet_with_score = (self.fake_subnet, 50, 1.)

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

    def test_scores(self):
        # test property: scores
        data = [self.fake_subnet_with_score] * 2
        candidates = Candidates(data)
        self.assertEqual(candidates.scores, [1., 1.])

    def test_resources(self):
        # test property: resources
        data = [self.fake_subnet_with_resource] * 2
        candidates = Candidates(data)
        self.assertEqual(candidates.scores, [50., 50.])

    def test_subnets(self):
        # test property: subnets
        data = [self.fake_subnet_with_score] * 2
        candidates = Candidates(data)
        self.assertEqual(candidates.subnets, [self.fake_subnet] * 2)

    def test_append(self):
        # item is dict
        candidates = Candidates()
        candidates.append(self.fake_subnet)
        self.assertEqual(len(candidates), 1)
        # item is tuple
        candidates = Candidates()
        candidates.append(self.fake_subnet_with_score)
        self.assertEqual(len(candidates), 1)

    def test_insert(self):
        # item is dict
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.insert(1, self.fake_subnet)
        self.assertEqual(len(candidates), 2)
        # item is tuple
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.insert(1, self.fake_subnet_with_score)
        self.assertEqual(len(candidates), 2)

    def test_extend(self):
        # other is list
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.extend([self.fake_subnet])
        self.assertEqual(len(candidates), 2)
        # other is UserList
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.extend(UserList([self.fake_subnet_with_score]))
        self.assertEqual(len(candidates), 2)

    def test_set_resources(self):
        # test set_resources
        candidates = Candidates([self.fake_subnet_with_resource])
        candidates.set_resources(0, 49.1)
        self.assertEqual(candidates[0][1], 49.1)

    def test_set_score(self):
        # test set_score
        candidates = Candidates([self.fake_subnet_with_score])
        candidates.set_score(0, 0.5)
        self.assertEqual(candidates[0][1], 50, 0.5)
