import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def create_experiment_directory(experiment_name):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_directory = f"experiments/{timestamp}_{experiment_name}"
    os.makedirs(experiment_directory, exist_ok=True)
    return experiment_directory


def log_experiment_details(experiment_directory, experiment_vars):
    experiment_details = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **experiment_vars
    }
    with open(os.path.join(experiment_directory, "experiment_details.json"), "w") as f:
        json.dump(experiment_details, f, indent=4)


def save_model(experiment_directory, model):
    model_path = os.path.join(experiment_directory, "model.h5")
    model.save(model_path)


def log_training_history(experiment_directory, history):
    history_file = os.path.join(experiment_directory, 'training_history.json')
    with open(history_file, 'w') as file:
        json.dump(history, file, indent=4)

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(experiment_directory, 'loss_plot.png'))
    plt.close()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Training Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(experiment_directory, 'accuracy_plot.png'))
    plt.close()


def log_classification_report(experiment_directory, y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=1)
    report_path = os.path.join(experiment_directory, "classification_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)


def log_confusion_matrix(experiment_directory, y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 10))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Normalized Confusion Matrix')
    plt.savefig(os.path.join(experiment_directory, 'confusion_matrix_normalized.png'))
    plt.close()


def plot_precision_recall_curve(experiment_directory, y_true, y_scores, n_classes):
    precision = dict()
    recall = dict()
    threshold = dict()
    for i in range(n_classes):
        precision[i], recall[i], threshold[i] = precision_recall_curve(y_true == i, y_scores[:, i])

    plt.figure(figsize=(7, 8))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], lw=2, label=f'Class {i}')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall Curve to Multi-Class')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(experiment_directory, 'precision_recall_curve.png'))
    plt.close()


def plot_class_probability_distributions(experiment_directory, y_scores, n_classes):
    plt.figure(figsize=(10, 6))
    for i in range(n_classes):
        sns.kdeplot(y_scores[:, i], label=f'Class {i}', shade=True)

    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Predicted Probability Distributions for Each Class')
    plt.legend()
    plt.savefig(os.path.join(experiment_directory, 'class_probability_distributions.png'))
    plt.close()


def plot_classification_report(experiment_directory, y_true, y_pred, target_names):
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    fig, ax = plt.subplots(figsize=(10, 7))
    report_df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')[
        ['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Classification Report')
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.savefig(os.path.join(experiment_directory, 'classification_report.png'))
    plt.close()