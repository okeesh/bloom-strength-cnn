import os
import numpy as np
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from numpy import argmax
from load_data import train_images, train_labels, validation_images, validation_labels, test_images, test_labels
from experiment_logger import create_experiment_directory, log_experiment_details, save_model, log_training_history, \
    log_classification_report, log_confusion_matrix, plot_class_probability_distributions, \
    plot_classification_report
from experiment_logger import plot_precision_recall_curve
import tensorflow as tf
from tensorflow.keras import backend as K

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)  # Convert y_true to float32
        y_pred = tf.cast(y_pred, tf.float32)  # Convert y_pred to float32
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

# Define the hyperparameters
learning_rate = 0.001
dropout_rate = 0.7
target_names = [f'class_{i}' for i in range(9)]
alpha = 0.25
gamma = 1.0
batch_size = 32
epochs = 50

print(np.unique(train_labels))
print(np.unique(validation_labels))
print(np.unique(test_labels))

# Check if the train, validation, and test labels have 9 unique classes
assert len(np.unique(train_labels)) == 9
assert len(np.unique(validation_labels)) == 9
assert len(np.unique(test_labels)) == 9

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classifier layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(dropout_rate)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dense(9, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Define the experiment name
experiment_name = f"{type(base_model).__name__}"

# Create a directory for the experiment
experiment_directory = create_experiment_directory(experiment_name)

experiment_vars = {
    "base_model": type(base_model).__name__,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "dropout_rate": dropout_rate,
    "alpha": alpha,
    "gamma": gamma,
    "model_layers": [layer.__class__.__name__ for layer in model.layers],
    "model_layers_count": len(model.layers),
    "model_summary": model.summary()
}

log_experiment_details(experiment_directory, experiment_vars)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with focal loss
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss=focal_loss(gamma=gamma, alpha=alpha),
              metrics=['accuracy'])

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(experiment_directory, 'checkpoints', 'model.h5'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model with validation data
history = model.fit(
    train_images,
    train_labels,
    validation_data=(validation_images, validation_labels),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[early_stopping, checkpoint]
)

print(history.history.keys())

# Predict the classes using the trained model
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=-1)

# Log the confusion matrix
log_confusion_matrix(experiment_directory, test_labels, y_pred_classes, target_names)

# Log the classification report
log_classification_report(experiment_directory, test_labels, y_pred_classes, target_names)

# Save the raw data
np.save(os.path.join(experiment_directory, "y_true_classes.npy"), test_labels)
np.save(os.path.join(experiment_directory, "y_pred_classes.npy"), y_pred_classes)

# Predict the probabilities
y_scores = model.predict(test_images)

# Call the functions
plot_precision_recall_curve(experiment_directory, test_labels, y_scores, len(target_names))
plot_class_probability_distributions(experiment_directory, y_scores, len(target_names))
plot_classification_report(experiment_directory, test_labels, y_pred_classes, target_names)

# Log the training history
log_training_history(experiment_directory, history.history)