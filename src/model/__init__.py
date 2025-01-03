from .classification import MangoClassificationModel as mango
from .regression import MangoRegression as mango_reg
from .tree import MangoTree as mango_tree
from .forest import MangoForest as mango_forest
from .neural import MangoNetwork as mango_net
from .dnn import MangoDeepNetwork as mango_dnn

__all__ = ("mango", "mango_reg", "mango_tree", "mango_forest", "mango_net", "mango_dnn")
