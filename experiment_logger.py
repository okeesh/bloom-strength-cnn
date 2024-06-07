import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
import matplotlib.pyplot as plt
import numpy as np


def create_experiment_directory(experiment_name):
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    experiment_directory = f"experiments/{timestamp}-{experiment_name}"
    os.makedirs(experiment_directory, exist_ok=True)
    return experiment_directory


def plot_heatmap(learning_rates, dropout_rates, accuracies):
    plt.figure(figsize=(10, 6))
    plt.imshow(accuracies, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Validation Accuracy')
    plt.xticks(np.arange(len(dropout_rates)), [str(rate) for rate in dropout_rates])
    plt.yticks(np.arange(len(learning_rates)), [str(rate) for rate in learning_rates])
    plt.xlabel('Dropout Rate')
    plt.ylabel('Learning Rate')
    plt.title('Validation Accuracy for each combination of Learning Rate and Dropout Rate')
    plt.show()


def log_experiment_details(experiment_directory, experiment_vars):
    experiment_details = {
        "timestamp": datetime.now().strftime("%m-%d %H:%M"),
        **experiment_vars
    }
    with open(os.path.join(experiment_directory, "experiment_details.json"), "w") as f:
        json.dump(experiment_details, f, indent=4)


def save_model(experiment_directory, model):
    model_path = os.path.join(experiment_directory, "model.h5")
    model.save(model_path)


def log_evaluation_metrics(experiment_directory, metrics):
    with open(os.path.join(experiment_directory, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def log_training_history(experiment_directory, history):
    # Save the validation loss and accuracy and training loss and accuracy
    training_history = {
        "val_loss": history.history['val_loss'],
        "val_accuracy": history.history['val_accuracy'],
        "loss": history.history['loss'],
        "accuracy": history.history['accuracy']
    }
    with open(os.path.join(experiment_directory, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=4)
    return training_history


def log_classification_report(experiment_directory, y_true, y_pred, target_names):
    print(classification_report(y_true, y_pred, target_names=target_names))
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    with open(f'{experiment_directory}/classification_report.json', 'w') as f:
        json.dump(report, f)


def log_confusion_matrix(experiment_directory, y_true, y_pred, class_names):
    # Generate the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap of the confusion matrix
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(label='Number of instances')
    plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
    plt.yticks(np.arange(len(class_names)), class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix')

    # Save the plot as a PNG file
    plt.savefig(os.path.join(experiment_directory, 'confusion_matrix.png'))

    # Close the plot to free up memory
    plt.close()
