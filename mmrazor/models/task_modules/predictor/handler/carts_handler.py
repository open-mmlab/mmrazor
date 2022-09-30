# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from mmrazor.registry import TASK_UTILS
from .base_handler import BaseHandler


@TASK_UTILS.register_module()
class CartsHandler(BaseHandler):
    """Classification and Regression Tree.

    Args:
        n_tree (int): number of regression trees.
    """

    def __init__(self, n_tree=1000):
        self.n_tree = n_tree
        self.model = None

    @staticmethod
    def _make_decision_trees(train_data, train_label, n_tree):
        """Construct the decision trees."""
        feature_record = []
        tree_record = []

        for i in range(n_tree):
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

        return tree_record, feature_record

    def fit(self, train_data, train_label):
        """Training predictor."""
        self.model = self._make_decision_trees(train_data, train_label,
                                               self.n_tree)

    def predict(self, test_data):
        """Predict the subnets' performance."""
        trees, features = self.model[0], self.model[1]
        test_num, n_tree = len(test_data), len(trees)

        predict_labels = np.zeros((test_num, 1))
        for i in range(test_num):
            this_test_data = test_data[i, :]
            predict_this_list = np.zeros(n_tree)

            for j, (tree, feature) in enumerate(zip(trees, features)):
                predict_this_list[j] = tree.predict([this_test_data[feature]
                                                     ])[0]

            predict_this_list = np.sort(predict_this_list)
            predict_this_list = predict_this_list[::-1]
            this_predict = np.mean(predict_this_list)
            predict_labels[i, 0] = this_predict

        return predict_labels
