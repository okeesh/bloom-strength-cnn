import tensorflow as tf
tf.config.run_functions_eagerly(True)

from keras.applications import MobileNetV2, ResNet50, VGG16

from dataset.load_data import load_data
from model_class.model import ModelTrainer
from model_class.model_config import ModelConfig

# Define input shape
input_shape = (224, 224, 3)

config = ModelConfig(
    model_type='hierarchical',
    learning_rate=0.001,
    dropout_rate=0.7,
    batch_size=32,
    epochs=50,
    pretrained_model=MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
    dense_units=256,
    num_classes=9,
    input_shape=input_shape  # Add this line to explicitly set input_shape
)

model = ModelTrainer(config)

# Load train and validation data from load_data
train_images, train_labels, validation_images, validation_labels, class_weights = load_data(model_type='hierarchical')
train_data = (train_images, train_labels)  # Include class_weights in train_data
validation_data = (validation_images, validation_labels)

# Train the model
history = model.train(train_data, validation_data)

# Optionally, you can print or plot the training history
print(history.history)