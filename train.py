import tensorflow as tf
tf.config.run_functions_eagerly(True)

from keras.applications import MobileNet, ResNet50, VGG16, MobileNetV2

from dataset.load_data import load_data
from model_class.model import ModelTrainer
from model_class.model_config import ModelConfig

# Define input shape
input_shape = (224, 224, 3)

config_classification = ModelConfig(
    model_type='classification',
    learning_rate=0.001,
    dropout_rate=0.5,
    batch_size=32,
    epochs=100,
    pretrained_model=MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
    dense_units=256,
    num_classes=9,
    input_shape=input_shape
)

# Configuration for Hierarchical Model
config_hierarchical = ModelConfig(
    model_type='hierarchical',
    learning_rate=0.001,
    dropout_rate=0.1,
    batch_size=16,
    epochs=10,
    pretrained_model=MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
    dense_units=512,
    num_classes=9,
    input_shape=input_shape
)

config_regression = ModelConfig(
    model_type='regression',
    learning_rate=0.01,
    activation="linear",
    regularization_rate = 0.0005,
    dropout_rate=0.1,
    batch_size=16,
    epochs=50,
    pretrained_model=MobileNet(weights='imagenet', include_top=False, input_shape=input_shape),
    dense_units=1024,
    input_shape=input_shape
)

model = ModelTrainer(config_regression)

# Load train and validation data from load_data
train_images, train_labels, validation_images, validation_labels, class_weights = load_data(model_type='regression')
train_data = (train_images, train_labels)
validation_data = (validation_images, validation_labels)

# Train the model
history = model.train(train_data, validation_data)

# Optionally, you can print or plot the training history
print(history.history)
