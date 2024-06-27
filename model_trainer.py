import os
import numpy as np
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from load_data import train_images, train_labels, validation_images, validation_labels, class_weights
from experiment_logger import create_experiment_directory, log_experiment_details, save_model, log_training_history, \
    log_classification_report, log_confusion_matrix, plot_precision_recall_curve, plot_class_probability_distributions, \
    plot_classification_report, plot_class_weights_and_frequencies
from keras.applications import MobileNet

# Define the hyperparameters
learning_rate = 0.001
dropout_rate = 0.7
target_names = [f'class_{i}' for i in range(9)]
batch_size = 32
epochs = 50

# Check if the train and validation labels have 9 unique classes
assert len(np.unique(np.argmax(train_labels, axis=1))) == 9
assert len(np.unique(np.argmax(validation_labels, axis=1))) == 9

print("Length of train_labels: ", len(train_labels))
print("Length of validation_labels: ", len(validation_labels))

# Load the pre-trained MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False)


# Add custom classifier layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(dropout_rate)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dense(9, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Define the experiment name
experiment_name = f"{type(base_model).__name__}"

# Create a directory for the experiment
experiment_directory = create_experiment_directory(experiment_name)

# Save the model architecture to a file in the experiment directory
experiment_vars = {
    "base_model": type(base_model).__name__,
    "learning_rate": learning_rate,
    "batch_size": batch_size,
    "epochs": epochs,
    "dropout_rate": dropout_rate,
    "class_weights": class_weights,
    "model_layers": [layer.__class__.__name__ for layer in model.layers],
    "model_layers_count": len(model.layers),
}

# Log the experiment details to a file in the experiment directory for future reference
log_experiment_details(experiment_directory, experiment_vars)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model with weighted cross-entropy loss
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
checkpoint = ModelCheckpoint(
    filepath=os.path.join(experiment_directory, 'checkpoints', 'model.h5'),
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Train the model with validation data and class weights for imbalanced data
history = model.fit(
    train_images,
    train_labels,
    validation_data=(validation_images, validation_labels),
    epochs=epochs,
    batch_size=batch_size,
    verbose=1,
    callbacks=[early_stopping, checkpoint],
    class_weight=class_weights
)

# Plot the class weights and their frequencies
plot_class_weights_and_frequencies(experiment_directory, class_weights, train_labels)

# Log the training history and save the model
log_training_history(experiment_directory, history.history)