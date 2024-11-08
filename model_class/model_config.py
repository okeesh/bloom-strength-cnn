import json
import pickle
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from keras.callbacks import Callback
from keras.models import Model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras import backend as K



@dataclass
class ModelConfig:
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    learning_rate: float = 0.001
    dropout_rate: float = 0.7
    batch_size: int = 32
    epochs: int = 50
    dense_units: int = 256
    regularization_rate: float = 0.001
    model_type: str = 'regression'
    pretrained_model: Optional[Model] = None
    activation: str = 'relu'
    num_classes: Optional[int] = None
    loss_function: Optional[str] = None

    def __post_init__(self):
        self.validate()

    def validate(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.dropout_rate <= 1:
            raise ValueError("dropout_rate must be between 0 and 1")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.model_type not in ['regression', 'classification', 'hierarchical', 'regression_normalized']:
            raise ValueError("model_type must be 'regression' or 'regression_normalized' or 'classification' or 'hierarchical'")
        if self.model_type == 'classification' and self.num_classes is None:
            raise ValueError("num_classes must be specified for classification models")

class LearningRateScheduler(Callback):
    def __init__(self, new_lr, change_epoch):
        super().__init__()
        self.new_lr = new_lr
        self.change_epoch = change_epoch

    def on_epoch_begin(self, epoch, logs=None):
        if epoch == self.change_epoch:
            K.set_value(self.model.optimizer.lr, self.new_lr)
            print(f"\nEpoch {epoch}: Lernrate auf {self.new_lr} geändert.")


class BaseModel(ABC):
    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def compile_model(self, model, class_weights=None):
        pass

    @abstractmethod
    def train_model(self, model, train_data, validation_data):
        pass

    def save_results(self, model, history, predictions, true_labels, experiment_directory):
        results = {
            'model_name': self.config.pretrained_model.name,
            'model_type': self.config.model_type,
            'hyperparameters': {
                'learning_rate': self.config.learning_rate,
                'regularization_rate': self.config.regularization_rate,
                'dropout_rate': self.config.dropout_rate,
                'batch_size': self.config.batch_size,
                'epochs': self.config.epochs,
                'activation': self.config.activation,
                # Add any other relevant hyperparameters
            },
            'history': {key: np.array(value).tolist() for key, value in history.history.items()},
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'true_labels': true_labels.tolist() if isinstance(true_labels, np.ndarray) else true_labels,
        }

        # Save as JSON
        with open(f"{experiment_directory}/results.json", 'w') as f:
            json.dump(results, f, indent=4)

        # Save predictions and true labels as pickle for easier loading later
        with open(f"{experiment_directory}/predictions_labels.pkl", 'wb') as f:
            pickle.dump({'predictions': predictions, 'true_labels': true_labels}, f)

        # Save model if needed
        model.save(f"{experiment_directory}/model.h5")

        print(f"Results saved in {experiment_directory}")

    def plot_confusion_matrix(self, y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Konfusionsmatrix')
        plt.ylabel('Wahres Label')
        plt.xlabel('Vorhergesagtes Label')
        plt.savefig(filename)
        plt.close()