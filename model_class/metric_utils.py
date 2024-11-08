import json
import os
from datetime import datetime

import numpy as np
from keras.losses import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def calculate_metrics(results_path):
    """
    Calculate metrics from results.json and create a new metrics file.

    Args:
        results_path (str): Path to the directory containing results.json
    """
    # Construct full path to results.json
    json_path = os.path.join(results_path, 'results.json')

    # Check if file exists
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"No results.json found in {results_path}")

    # Load the JSON file
    with open(json_path, 'r') as f:
        results = json.load(f)

    # Convert predictions and true labels to numpy arrays
    predictions = np.array(results['predictions'])
    true_labels = np.array(results['true_labels'])

    # Get predicted and true classes
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(true_labels, axis=1)

    # Calculate metrics
    metrics = {
        'balanced_accuracy': float(balanced_accuracy_score(true_classes, predicted_classes)),
        'accuracy': float(accuracy_score(true_classes, predicted_classes)),
        'mse': float(mean_squared_error(true_classes, predicted_classes)),
        'mae': float(mean_absolute_error(true_classes, predicted_classes))
    }

    # Get model info if available
    model_info = {}
    for key in ['model_name', 'model_type', 'hyperparameters']:
        if key in results:
            model_info[key] = results[key]

    # Create metrics filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_filename = f"metrics_{timestamp}.txt"
    metrics_path = os.path.join(results_path, metrics_filename)

    # Write metrics to file
    with open(metrics_path, 'w') as f:
        # Write model info if available
        if model_info:
            f.write("Model Information:\n")
            f.write("=================\n")
            for key, value in model_info.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

        # Write metrics
        f.write("Metrics:\n")
        f.write("========\n")
        for metric_name, value in metrics.items():
            f.write(f"{metric_name}: {value:.4f}\n")

    return metrics, metrics_path


def process_experiment_folders(base_directory):
    """
    Process all experiment folders in the base directory.

    Args:
        base_directory (str): Base directory containing experiment folders

    Returns:
        list: List of tuples containing (directory_path, metrics)
    """
    results = []

    # Get immediate subdirectories
    experiment_folders = [d for d in os.listdir(base_directory)
                          if os.path.isdir(os.path.join(base_directory, d))]

    print(f"Found {len(experiment_folders)} experiment folders")

    for folder in experiment_folders:
        folder_path = os.path.join(base_directory, folder)
        try:
            metrics, metrics_file = calculate_metrics(folder_path)
            results.append((folder_path, metrics))
            print(f"\nProcessed {folder}")
            print(f"Metrics file created: {metrics_file}")
            print("Metrics:")
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        except FileNotFoundError:
            print(f"\nNo results.json found in {folder}")
        except Exception as e:
            print(f"\nError processing {folder}: {str(e)}")

    return results


def create_summary_file(base_directory, results):
    """
    Create a summary file with metrics from all experiments.

    Args:
        base_directory (str): Base directory
        results (list): List of (directory_path, metrics) tuples
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(base_directory, f"summary_{timestamp}.txt")

    with open(summary_path, 'w') as f:
        f.write("Experiments Summary\n")
        f.write("==================\n\n")

        for directory, metrics in results:
            f.write(f"Directory: {os.path.basename(directory)}\n")
            f.write("-" * 50 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python script.py <experiments_directory>")
        sys.exit(1)

    directory = sys.argv[1]
    try:
        # Process all experiment folders
        results = process_experiment_folders(directory)

        # Create summary file
        if results:
            create_summary_file(directory, results)
            print("\nProcessing complete! Check individual metrics files in each folder")
            print("and the summary file in the base directory.")
        else:
            print("\nNo results were processed.")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)