import json
import os
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import time


def print_header():
    print("\n" + "=" * 50)
    print("🔍 Bloom Strength Evaluation Tool".center(50))
    print("=" * 50 + "\n")


def print_success(text):
    print(f"\n✅ {text}")


def print_error(text):
    print(f"\n❌ {text}")


def loading_animation(duration=1.5):
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()
    i = 0
    while time.time() - start_time < duration:
        print(f"\r{animation[i]} Processing...", end="")
        time.sleep(0.1)
        i = (i + 1) % len(animation)
    print("\r" + " " * 20 + "\r", end="")


def print_menu():
    print("\nChoose an action:")
    print("-" * 20)
    print("1. 📊 Calculate All Metrics")
    print("2. 🔙 Go Back")
    print("3. 🚪 Exit")
    print("-" * 20)


def calculate_metrics(folder_name):
    """
    Calculate MAE, MSE, Accuracy, Balanced Accuracy, and Cumulated Accuracy metrics from the JSON file.
    """
    # Read the JSON file
    file_path = os.path.join('evaluate', folder_name, 'results.json')
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get predictions and true labels from the JSON data
    predictions = np.array(data['predictions'])
    true_labels = np.array(data['true_labels'])
    model_type = data['model_type']

    # Handle different model types
    if model_type in ['hierarchical', 'classification']:
        # Get predicted and true classes for standard metrics
        pred_classes = np.argmax(predictions, axis=1) + 1
        true_classes = np.argmax(true_labels, axis=1) + 1

        # Calculate standard metrics
        mae = np.mean(np.abs(pred_classes - true_classes))
        mse = np.mean((pred_classes - true_classes) ** 2)
        accuracy = np.mean(pred_classes == true_classes)
        balanced_acc = balanced_accuracy_score(true_classes, pred_classes)

        # Calculate cumulated accuracy using the hierarchical approach
        mask_wide = (true_labels > 0).astype(np.float32)  # Create mask for valid positions
        truths = np.argmax(mask_wide * (predictions + 1), axis=1)
        preds = np.argmax(predictions, axis=1)
        cumulated_acc = np.mean(truths == preds)

    else:  # regression
        # Flatten arrays properly - predictions are in shape (n, 1)
        pred_values = np.array([p[0] for p in predictions])
        true_values = np.array([t[0] for t in true_labels])

        # Calculate regression metrics directly on raw values
        mae = np.mean(np.abs(pred_values - true_values))
        mse = np.mean((pred_values - true_values) ** 2)

        # For accuracy, we need to round predictions
        rounded_preds = np.round(pred_values)
        accuracy = np.mean(rounded_preds == true_values)

        # For balanced accuracy with regression
        unique_classes = np.unique(true_values)
        class_accuracies = []
        for cls in unique_classes:
            mask = (true_values == cls)
            if np.sum(mask) > 0:
                class_accuracy = np.mean(rounded_preds[mask] == cls)
                class_accuracies.append(class_accuracy)
        balanced_acc = np.mean(class_accuracies)

        # For regression, we'll still use the ±1 criterion for cumulated accuracy
        diff = np.abs(rounded_preds - true_values)
        cumulated_acc = np.mean(diff <= 1)

    # Create metrics directory
    metrics_dir = os.path.join('evaluate', folder_name, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Save metrics
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'ACC': accuracy,
        'BAC': balanced_acc,
        'CAC': cumulated_acc
    }

    # Save all metrics to their respective files
    for metric_name, metric_value in metrics.items():
        filename = os.path.join(metrics_dir, f'{metric_name}_{metric_value:.2f}.txt')
        with open(filename, 'w') as f:
            f.write(f'{metric_name}: {metric_value:.4f}\n')
            f.write(f'Model Type: {model_type}\n')
            if model_type == 'regression':
                f.write('Note: For accuracy calculations, predictions were rounded to nearest integer\n')
            f.write(f'Number of samples: {len(predictions)}')

    print(f"\nMetrics saved in {metrics_dir}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Cumulated Accuracy: {cumulated_acc:.4f}")

    return metrics


def select_folder():
    print("\nChoose the folder to evaluate:")
    # Get all folders in ./evaluate
    folders = [f for f in os.listdir('evaluate') if os.path.isdir(os.path.join('evaluate', f))]

    if not folders:
        print_error("No folders found in 'evaluate' directory!")
        exit(1)

    for i, folder in enumerate(folders):
        print(f"{i + 1}: {folder}")

    while True:
        try:
            number = int(input("\n📂 Enter folder number: "))
            if 1 <= number <= len(folders):
                folder_name = folders[number - 1]
                print_success(f"Selected: {folder_name}")
                return folder_name
            print_error(f"Please enter a number between 1 and {len(folders)}")
        except ValueError:
            print_error("Please enter a valid number")


if __name__ == '__main__':
    print_header()

    # Check if evaluate directory exists
    if not os.path.exists('evaluate'):
        print_error("'evaluate' directory not found!")
        exit(1)

    while True:
        folder_name = select_folder()

        while True:
            print_menu()
            choice = input("👉 Enter your choice: ")

            if choice == '1':
                print("\n🔄 Processing metrics...")
                loading_animation()
                try:
                    calculate_metrics(folder_name)
                    print_success("Metrics calculated and saved successfully!")
                    print("\n" + "=" * 50)
                except Exception as e:
                    print_error(f"An error occurred: {str(e)}")
            elif choice == '2':
                print("\n🔙 Going back to folder selection...")
                break
            elif choice == '3':
                print("\n👋 Goodbye!")
                exit(0)
            else:
                print_error("Please enter 1, 2, or 3")