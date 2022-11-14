# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from mmrazor.registry import TASK_UTILS
from .base_handler import BaseHandler


@TASK_UTILS.register_module()
class CartsHandler(BaseHandler):
    """Classification and Regression Tree.

    Args:
        num_trees (int): number of regression trees.
    """

    def __init__(self, num_trees=1000):
        self.num_trees = num_trees

    def fit(self, train_data: np.array, train_label: np.array) -> None:
        """Define the model of handler.

        Args:
            train_data (numpy.array): input data for training.
            train_label (numpy.array): input label for training.
        """
        self.model = self._make_decision_trees(train_data, train_label,
                                               self.num_trees)

    def predict(self, test_data: np.array) -> np.array:
        """Predict the evaluation metric of the model.

        Args:
            test_data (numpy.array): input data for testing.

        Returns:
            numpy.array: predicted metric.
        """
        trees, features = self.model[0], self.model[1]
        test_num, num_trees = len(test_data), len(trees)

        predict_labels = np.zeros((test_num, 1))
        for i in range(test_num):
            this_test_data = test_data[i, :]
            predict_this_list = np.zeros(num_trees)

            for j, (tree, feature) in enumerate(zip(trees, features)):
                predict_this_list[j] = tree.predict([this_test_data[feature]
                                                     ])[0]

            predict_this_list = np.sort(predict_this_list)
            predict_this_list = predict_this_list[::-1]
            this_predict = np.mean(predict_this_list)
            predict_labels[i, 0] = this_predict

        return predict_labels

    @staticmethod
    def _make_decision_trees(train_data: np.array, train_label: np.array,
                             num_trees: int) -> List[list]:
        """Construct the decision trees.

        Args:
            train_data (numpy.array): input data for training.
            train_label (numpy.array): input label for training.
            num_trees (int): num of decision trees.

        Returns:
            List[list]: List of built models.
        """
        feature_record = []
        tree_record = []

        for _ in range(num_trees):
            sample_idx = np.arange(train_data.shape[0])
            np.random.shuffle(sample_idx)
            train_data = train_data[sample_idx, :]
            train_label = train_label[sample_idx]

            feature_idx = np.arange(train_data.shape[1])
            np.random.shuffle(feature_idx)
            n_feature = np.random.randint(1, train_data.shape[1] + 1)
            selected_feature_ids = feature_idx[0:n_feature]
            feature_record.append(selected_feature_ids)

            dt = DecisionTreeRegressor()
            dt.fit(train_data[:, selected_feature_ids], train_label)
            tree_record.append(dt)

        return [tree_record, feature_record]
