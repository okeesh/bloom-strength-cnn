import copy

from models.classification_model import ClassificationModel
from models.hierarchicalpartial_model import HierarchicalPartialLossModel
from model_class.model_config import ModelConfig
from models.regression_model import RegressionModel

import itertools
from typing import Dict, List, Any, Tuple


class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = self._get_model()

    def _get_model(self):
        if self.config.model_type == 'regression':
            return RegressionModel(self.config)
        elif self.config.model_type == 'classification':
            return ClassificationModel(self.config)
        elif self.config.model_type == 'hierarchical':
            return HierarchicalPartialLossModel(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def train(self, train_data, validation_data):
        model = self.model.create_model()
        compiled_model = self.model.compile_model(model)
        return self.model.train_model(compiled_model, train_data, validation_data)


