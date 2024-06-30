from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional
from keras.models import Model


@dataclass
class ModelConfig:
    learning_rate: float = 0.001
    dropout_rate: float = 0.7
    batch_size: int = 32
    epochs: int = 50
    dense_units: int = 256
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
        if self.model_type not in ['regression', 'classification']:
            raise ValueError("model_type must be 'regression' or 'classification'")
        if self.model_type == 'classification' and self.num_classes is None:
            raise ValueError("num_classes must be specified for classification models")


class BaseModel(ABC):
    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def compile_model(self, model):
        pass

    @abstractmethod
    def train_model(self, model, train_data, validation_data):
        pass
