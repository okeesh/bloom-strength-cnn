import os
import numpy as np
from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from load_data import train_images, train_labels, validation_images, validation_labels, test_images, test_labels, class_weights
from experiment_logger import create_experiment_directory, log_experiment_details, save_model, log_training_history, \
    log_classification_report, log_confusion_matrix, plot_precision_recall_curve, plot_class_probability_distributions, \
    plot_classification_report, plot_class_weights_and_frequencies

# Define the hyperparameters
learning_rate = 0.001
dropout_rate = 0.7
target_names = [f'class_{i}' for i in range(9)]
batch_size = 32
epochs = 50

# Check if the train, validation, and test labels have 9 unique classes
assert len(np.unique(np.argmax(train_labels, axis=1))) == 9
assert len(np.unique(np.argmax(validation_labels, axis=1))) == 9
assert len(np.unique(np.argmax(test_labels, axis=1))) == 9

print("Length of train_labels: ", len(train_labels))
print("Length of validation_labels: ", len(validation_labels))
print("Length of test_labels: ", len(test_labels))

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

# Predict the classes using the trained model on the test data
y_pred = model.predict(test_images)
y_pred_classes = np.argmax(y_pred, axis=-1)

# Convert test_labels from one-hot encoding to class labels for classification report and confusion matrix
y_true_classes = np.argmax(test_labels, axis=1)

# Log the confusion matrix for the test data predictions and true labels
log_confusion_matrix(experiment_directory, y_true_classes, y_pred_classes, target_names)

# Log the classification report for the test data predictions and true labels
log_classification_report(experiment_directory, y_true_classes, y_pred_classes, target_names)

# Save the raw data for further analysis if needed
np.save(os.path.join(experiment_directory, "y_true_classes.npy"), test_labels)
np.save(os.path.join(experiment_directory, "y_pred_classes.npy"), y_pred_classes)


# -----------------------------------
# Additional code for logging metrics and visualizations after training the model

# Predict the probabilities
y_scores = model.predict(test_images)

# Call the functions to log the metrics and visualizations
plot_precision_recall_curve(experiment_directory, y_true_classes, y_scores, len(target_names))
plot_class_probability_distributions(experiment_directory, y_scores, len(target_names))
plot_classification_report(experiment_directory, y_true_classes , y_pred_classes, target_names)

# Log the training history and save the model
log_training_history(experiment_directory, history.history)