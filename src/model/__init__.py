from .decision import MangoTree as decision_tree
from .regression import MangoRegression as mv_regression
from .forest import MangoForest as random_forest
from .deep import DeepMangoNetwork as deep_neural_network
from .neural import MangoNetwork as neural_network

__all__ = (
    "decision_tree",
    "mv_regression",
    "random_forest",
    "deep_neural_network",
    "neural_network",
)
