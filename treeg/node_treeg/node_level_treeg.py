from treeg.node_treeg.tree_node_learner_node_level import TreeNodeLearner, TreeNodeLearnerParams
from treeg.node_treeg.graph_data_node_level import GraphData
from typing import List
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from collections import defaultdict


################################################################
###                                                          ###
### Note that line 124 of boosting.py in starboost should    ###
### be changed to:                                           ###
###   y_pred[:, i] += self.learning_rate * direction[:, i]   ###
###                                                          ###
### The same fix should be applied in line 179               ###
### /usr/local/lib/python3.8/dist-packages/starboost/        ###
###                                                          ###
################################################################

class NodeTreeG(BaseEstimator, RegressorMixin):
    def __init__(self,
                 graph: GraphData = None,
                 walk_lens: List[int] = [0, 1, 2],
                 max_attention_depth: int = 2,
                 max_number_of_leafs: int = 10,
                 min_gain: float = 0.0,
                 min_leaf_size: int = 10,
                 attention_types: List[int] = [1, 4],
                 attention_type_sample_probability: float = 0.5,
                 random_state: int = 0,
                 ):
        self.node_count = None
        self.tree_depth = None
        self.train_total_gain = None
        self.train_L2 = None
        self.trained_tree_ = None
        self.tree_learner_root_ = None
        self.walk_lens = walk_lens
        self.max_attention_depth = max_attention_depth
        self.max_number_of_leafs = max_number_of_leafs
        self.min_gain = min_gain
        self.min_leaf_size = min_leaf_size
        self.graph = graph
        self.attention_types = attention_types
        self.attention_type_sample_probability = attention_type_sample_probability
        self.stats_dict = None,
        self.feature_importances_ = None

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained sub-objects that are estimators.
        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as :class:`~sklearn.pipeline.Pipeline`). The latter have
        parameters of the form ``<component>__<parameter>`` so that it's
        possible to update each component of a nested object.
        Parameters
        ----------
        **params : dict
            Estimator parameters.
        Returns
        -------
        self : estimator instance
            Estimator instance.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].treeg_params(**sub_params)

        return self

    def fit(self, X: np.array, y: np.array):
        X = X.flatten()
        self.graph.compute_walks(self.walk_lens[-1])

        params = TreeNodeLearnerParams(
            walk_lens=self.walk_lens,
            max_attention_depth=self.max_attention_depth,
            max_number_of_leafs=self.max_number_of_leafs,
            min_gain=self.min_gain,
            min_leaf_size=self.min_leaf_size,
            graph=self.graph,
            attention_types=self.attention_types,
            attention_type_sample_probability=self.attention_type_sample_probability,
        )
        self.tree_learner_root_ = TreeNodeLearner(params=params, active=np.array(X),
                                                  parent=None)
        self.train_L2, self.train_total_gain, self.stats_dict = self.tree_learner_root_.fit(X, y)
        self.node_count, self.tree_depth = self.tree_learner_root_.node_count, self.tree_learner_root_.tree_depth
        self.feature_importances_ = self.compute_feature_importances(self.graph.get_number_of_features())
        return self

    def compute_feature_importances(self, num_of_features):
        cum_feature_gain = np.zeros(num_of_features)
        self.cum_feature_gain_traverse(self.tree_learner_root_, cum_feature_gain)
        for idx in range(num_of_features):
            if self.stats_dict['feature_index'][idx] > 0:
                cum_feature_gain[idx] /= self.stats_dict['feature_index'][idx]
        return cum_feature_gain

    def cum_feature_gain_traverse(self, trained_node, cum_feature_gain):
        if trained_node.gt is None:
            return
        cum_feature_gain[trained_node.feature_index] += trained_node.potential_gain
        self.cum_feature_gain_traverse(trained_node.gt, cum_feature_gain)
        self.cum_feature_gain_traverse(trained_node.lte, cum_feature_gain)

    def predict(self, X: List[int]):
        all_predictions = self.tree_learner_root_.predict_all()
        if isinstance(X, np.ndarray):
            X = X[0].tolist()
        array = np.array(all_predictions[X])
        return array.reshape(-1, 1)

    def print_tree(self):
        self.tree_learner_root_.print_tree()
