
import numpy as np
from collections import Counter

class Node:
	def __init__(self, feature=None, threshold=None, right=None, left=None, *,value=None):
		self.feature = feature
		self.threshold = threshold
		self.right = right
		self.left = left
		self.value = value

	def is_leaf_node(self):
		return self.value is not None

class Tree:
	def __init__(self, min_samples_split, max_depth, n_features=None):
		self.min_samples_split = min_samples_split
		self.max_depth = max_depth
		self.n_features =  n_features
		self.root = None



	def fit(self, X, y):

		self.n_features == X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)

		self.root = self._grow_tree(X, y)

	def _grow_tree(self, X, y, depth=0):
		n_samples, n_feats = X.shape
		n_labels = len(np.unique(y))


		#check stopping criteria
		if (depth>=self.max_depth or n_labels==1 or n_samples<=self.min_samples_split):
			leaf_value = self._most_common_label(y)
			return Node(value=leaf_value)


		feat_idx = np.random.choice(n_feats, self.n_features, replace=False)

		# find best split
		best_feature, best_thresh = self._best_split(X, y, feat_idx)



		# create child nodes
	def _best_split(self, X, y, feat_idxs):
		best_gain = -1
		split_idx, split_threshold = None, None

		for feat_idx in feat_idxs:
			X_column = X[:, feat_idx]
			thresholds = np.unique(X_column)

			for thr in thresholds:
				gain = self._information_gain(y, X_column, thr)

				if gain > best_gain:
					best_gain = gain
					split_idx = feat_idx
					split_threshold = thr

		return split_idx, split_threshold



	def _information_gain(self, y, X_Column, threshold):
		# parent entropy
		parent_entropy = self._entropy(y)


		# create children


		# calculate weighted entropy of childern


		# calculate information gain using above



		return

	def _entropy(self, y):


	def _most_common_label(self, y):
		counter = Counter(y)
		value = counter.most_common(1)[0][0]
		return value
	def predict(self):










