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

    def grid_search(self, param_grid: Dict[str, List[Any]], train_data: tuple, validation_data: tuple) -> Tuple[
        Dict[str, Any], float]:
        best_params = None
        best_score = float('inf') if self.config.model_type == 'regression' else float('-inf')

        # Generate all combinations of parameters
        param_combinations = [dict(zip(param_grid.keys(), values)) for values in
                              itertools.product(*param_grid.values())]

        for params in param_combinations:
            # Create a new config with the current parameters
            current_config = copy.deepcopy(self.config)
            for key, value in params.items():
                setattr(current_config, key, value)

            # Create a new model trainer with the current config
            current_trainer = ModelTrainer(current_config)

            # Train and evaluate the model
            history = current_trainer.train(train_data, validation_data)

            # Get the last validation score
            if self.config.model_type == 'regression':
                score = history.history['val_mse'][-1]
                is_better = score < best_score
            else:  # classification
                score = history.history['val_accuracy'][-1]
                is_better = score > best_score

            # Update best parameters if current score is better
            if is_better:
                best_score = score
                best_params = params

        # Update the config with the best parameters
        for key, value in best_params.items():
            setattr(self.config, key, value)

        return best_params, best_score


