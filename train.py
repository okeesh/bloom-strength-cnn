import tensorflow as tf
tf.config.run_functions_eagerly(True)

from keras.applications import MobileNet, ResNet50, VGG16

from dataset.load_data import load_data
from model_class.model import ModelTrainer
from model_class.model_config import ModelConfig

# Define input shape
input_shape = (224, 224, 3)

config = ModelConfig(
    model_type='hierarchical',
    learning_rate=0.001,  # Reduced learning rate
    dropout_rate=0.3,  # Slightly increased dropout
    batch_size=32,
    epochs=100,  # Increased number of epochs
    pretrained_model=MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3)),
    dense_units=1024,
    num_classes=9,
    input_shape=(224, 224, 3)
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