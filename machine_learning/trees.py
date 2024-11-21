import numpy as np
from typing import NoReturn, List
from collections import Counter

class Node:
    def __init__(self, gini, samples, value, feature_index=None, threshold=None, left=None, right=None) -> NoReturn:
        self.gini = gini
        self.samples = samples
        self.value = value
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTree:
    """
    Decision Tree classifier.

    Parameters
    ----------
    max_depth : int, optional
        The maximum depth of the tree. If None, the tree will grow until
        all leaves are pure or until all leaves contain less than two samples.
    """

    def __init__(self, max_depth=None) -> NoReturn:
        self.max_depth = max_depth
        self.root = None

    def gini_impurity(self, y):
        """
        Calculates the Gini impurity for a given set of labels.

        Parameters
        ----------
        y : array-like
            The labels of the samples.

        Returns
        -------
        gini : float
            The Gini impurity.
        """
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def split_data(self, X, y, feature_index, threshold) -> tuple:
        """
        Splits the data into two subsets based on the given feature index and threshold.

        Parameters
        ----------
        X : array-like
            The feature matrix.
        y : array-like
            The labels of the samples.
        feature_index : int
            The index of the feature to split on.
        threshold : float
            The threshold value.

        Returns
        -------
        X_left, y_left, X_right, y_right : tuple
            The left and right subsets of the data.
        """
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def best_split(self, X, y):
        """
        Finds the best split for the data.

        Parameters
        ----------
        X : array-like
            The feature matrix.
        y : array-like
            The labels of the samples.

        Returns
        -------
        best_feature : int
            The index of the feature that results in the best split.
        best_threshold : float
            The threshold value for the best split.
        """
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self.split_data(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gini_left = self.gini_impurity(y_left)
                gini_right = self.gini_impurity(y_right)
                weighted_gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        """
        Builds a decision tree.

        Parameters
        ----------
        X : array-like
            The feature matrix.
        y : array-like
            The labels of the samples.
        depth : int, optional
            The current depth of the tree.

        Returns
        -------
        node : Node
            The root of the tree.
        """
        num_samples = len(y)
        num_classes = len(np.unique(y))
        gini = self.gini_impurity(y)
        value = np.bincount(y).argmax()  # La classe con il maggior numero di occorrenze

        node = Node(gini=gini, samples=num_samples, value=value)

        # Fermarsi se la purezza è completa o si è raggiunta la profondità massima
        if gini == 0 or (self.max_depth is not None and depth >= self.max_depth):
            return node

        feature_index, threshold = self.best_split(X, y)
        if feature_index is None:
            return node

        X_left, y_left, X_right, y_right = self.split_data(X, y, feature_index, threshold)

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self.build_tree(X_left, y_left, depth + 1)
        node.right = self.build_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X, y):
        """
        Fits the tree to the data.

        Parameters
        ----------
        X : array-like
            The feature matrix.
        y : array-like
            The labels of the samples.
        """
        self.root = self.build_tree(X, y)

    def predict_sample(self, node, x):
        """
        Predicts the class of a single sample.

        Parameters
        ----------
        node : Node
            The current node.
        x : array-like
            The feature vector of the sample.

        Returns
        -------
        predicted_class : int
            The predicted class.
        """
        if node.left is None and node.right is None:
            return node.value

        if x[node.feature_index] < node.threshold:
            return self.predict_sample(node.left, x)
        else:
            return self.predict_sample(node.right, x)

    def predict(self, X):
        """
        Predicts the class of multiple samples.

        Parameters
        ----------
        X : array-like
            The feature matrix.

        Returns
        -------
        predicted_classes : array-like
            The predicted classes.
        """
        return np.array([self.predict_sample(self.root, x) for x in X])





class RandomForest:
    """
    Random Forest classifier.

    Parameters
    ----------
    size : int
        Number of decision trees in the forest.
    max_depth : int, optional
        Maximum depth of each tree.
    """
    def __init__(self, size, max_depth=None) -> NoReturn:
        self.size = size
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y) -> NoReturn:
        """
        Fits the random forest to the data using bootstrapping and feature subsampling.

        Parameters
        ----------
        X : array-like
            The feature matrix.
        y : array-like
            The labels of the samples.
        """
        n_samples, n_features = X.shape

        for _ in range(self.size):
            # Bootstrap sampling (with replacement)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[indices]
            y_bootstrap = y[indices]

            # Create a DecisionTree and fit it
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap)

            self.trees.append(tree)

    def predict(self, X):
        """
        Predicts the class of multiple samples using majority voting.

        Parameters
        ----------
        X : array-like
            The feature matrix.

        Returns
        -------
        predicted_classes : array-like
            The predicted classes.
        """
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Perform majority voting for each sample
        majority_votes = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]

        return np.array(majority_votes)


if __name__ == "__main__":
    print("Random Forest Example:")

    # Example data
    X = np.array([[2.7, 2.5], [1.3, 1.5], [3.1, 3.3], [1.2, 1.2], [300.2, 300.1]])
    y = np.array([0, 0, 1, 0, 2])

    # Create and train the Random Forest
    forest = RandomForest(size=3, max_depth=5)
    forest.fit(X, y)

    # Predictions
    samples = np.array([[2.5, 2.4], [1.3, 1.5], [3.2, 3.07], [1.2, 1.2], [300.2, 100.1]])
    predictions = forest.predict(samples)
    print("Random Forest Predictions:", predictions)
