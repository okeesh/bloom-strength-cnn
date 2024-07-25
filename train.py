from keras.applications import MobileNetV2

from dataset.load_data import load_data
from model_class.model import ModelTrainer
from model_class.model_config import ModelConfig

config = ModelConfig(
    model_type='hierarchical',
    learning_rate=0.001,
    dropout_rate=0.5,
    batch_size=16,
    epochs=10,
    pretrained_model=MobileNetV2(weights='imagenet', include_top=False),
    dense_units=256,
    num_classes=9
)

model = ModelTrainer(config)

# Load train and validation data from load_data
train_images, train_labels, validation_images, validation_labels, class_weights = load_data(model_type='hierarchical')
train_data = (train_images, train_labels)
validation_data = (validation_images, validation_labels)
model.train(train_data, validation_data)

