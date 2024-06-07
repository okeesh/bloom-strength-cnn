import os
import numpy as np
from keras.applications import ResNet50, MobileNet
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from numpy import argmax
from load_data import train_images, train_labels, test_images, test_labels
from experiment_logger import create_experiment_directory, log_experiment_details, save_model, log_training_history, \
    log_classification_report, log_confusion_matrix

# Define the hyperparameters
learning_rate = 0.001
dropout_rate = 0.7
target_names = [f'class_{i}' for i in range(9)]

# Load the pre-trained ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classifier layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(dropout_rate)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)  # Add L2 regularization to this layer
x = Dense(9, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 5
batch_size = 16

# Capture relevant variables
experiment_vars = {
    "base_model": type(base_model).__name__,
    # add the layers of the model that we added
    "model_layers": [
        layer.__class__.__name__ for layer in model.layers
    ],
    "model_layers_count": len(model.layers),
    "optimizer": type(optimizer).__name__,
    "learning_rate": learning_rate,
    "dropout_rate": dropout_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "model_summary": model.summary(),
    "train_samples": len(train_images),
    "test_samples": len(test_images),

}

# Create an experiment directory
experiment_name = f"{type(base_model).__name__}"
experiment_directory = create_experiment_directory(experiment_name)

# Log experiment details
log_experiment_details(experiment_directory, experiment_vars)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Define model checkpoint in the experiment directory
checkpoint = ModelCheckpoint(
    filepath=f"{experiment_directory}/checkpoints/model.h5",
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model
history = model.fit(
    train_images,
    train_labels,
    validation_data=(test_images, test_labels),
    epochs=epochs,
    callbacks=[early_stopping, checkpoint]  # Add early stopping
)

# # Unfreeze the top layers of the base model
# for layer in base_model.layers[-20:]:
#     layer.trainable = True
#
# # Recompile the model
# model.compile(optimizer=Adam(learning_rate=learning_rate/10), loss='categorical_crossentropy', metrics=['accuracy'])


# Continue training with early stopping and smaller learning rate
# history_fine_tuning = model.fit(
#     train_images,
#     train_labels,
#     validation_data=(test_images, test_labels),
#     epochs=epochs,
#     callbacks=[early_stopping, checkpoint]
# )

# Predict the classes
y_pred = model.predict(test_images)
y_pred_classes = argmax(y_pred, axis=-1)

# Convert one-hot encoded test labels to class labels
y_true_classes = argmax(test_labels, axis=-1)

# Log the confusion matrix
log_confusion_matrix(experiment_directory, y_true_classes, y_pred_classes, target_names)

# Log the classification report
log_classification_report(experiment_directory, y_true_classes, y_pred_classes, target_names)

# Save the raw data
np.save(os.path.join(experiment_directory, "y_true_classes.npy"), y_true_classes)
np.save(os.path.join(experiment_directory, "y_pred_classes.npy"), y_pred_classes)

# Save the trained model
# save_model(experiment_directory, model)

# Log the training history
log_training_history(experiment_directory, history)
