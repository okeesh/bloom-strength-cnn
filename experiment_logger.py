import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, auc



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
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=1)
    report_path = os.path.join(experiment_directory, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


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

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle



from sklearn.metrics import precision_recall_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle

def plot_precision_recall_curve(experiment_directory, y_true, y_scores, n_classes):
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])

    # Compute Precision-Recall and plot curve for each class
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_scores[:, i])
        average_precision[i] = auc(recall[i], precision[i])

    # Plot Precision-Recall curve for each class
    plt.figure(figsize=(7, 8))
    lines = []
    labels = []
    for i, color in zip(range(n_classes), cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(i, average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc='upper right', bbox_to_anchor=(1.3, 1))  # Adjust the position of the legend
    plt.savefig(os.path.join(experiment_directory, 'precision_recall_curve.png'))
    plt.close()


import seaborn as sns

def plot_class_probability_distributions(experiment_directory, y_scores, n_classes):
    # Plot the predicted probability distributions for each class
    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        sns.kdeplot(y_scores[:, i], label=f'Class {i}')

    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Predicted Probability Distributions for Each Class')
    plt.legend()
    plt.savefig(os.path.join(experiment_directory, 'class_probability_distributions.png'))
    plt.close()

import pandas as pd

def plot_classification_report(experiment_directory, y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    report_df[['precision', 'recall', 'f1-score']].drop(['accuracy', 'macro avg', 'weighted avg']).plot(kind='bar', figsize=(10, 7))
    plt.title('Classification Report')
    plt.grid(True)
    plt.savefig(os.path.join(experiment_directory, 'classification_report.png'))
    plt.close()