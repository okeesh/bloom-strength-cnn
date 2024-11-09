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


def get_standard_limits():
    """
    Define standard y-axis limits for each metric type based on actual value ranges
    """
    return {
        'mse': {
            'min': 1.0,  # Never below 1.5, but giving some padding
            'max': 5.0  # Never above 3.0
        },
        'mae': {
            'min': 0.5,  # Never below 0.8
            'max': 5.0  # Never above 2.0, giving some padding
        },
        'accuracy': {
            'min': 0.0,
            'max': 1.0  # Natural limits for accuracy
        }
    }


def plot_training_curves(folder_name):
    """
    Plot training and validation curves with optimized scales.
    """
    import matplotlib.pyplot as plt

    # Read the JSON file
    file_path = os.path.join('evaluate', folder_name, 'results.json')
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Create plots directory
    plots_dir = os.path.join('evaluate', folder_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Get standard limits
    limits = get_standard_limits()

    # Set style for all plots
    plt.style.use('seaborn')

    # Plot MSE and MAE
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # MSE plot
    ax1.plot(data['history']['loss'], label='Training MSE', color='blue')
    ax1.plot(data['history']['val_loss'], label='Validierung MSE', color='orange')
    ax1.set_title('Training und Validierung MSE')
    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('MSE')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(limits['mse']['min'], limits['mse']['max'])

    # Add minor gridlines for better readability given the smaller scale
    ax1.minorticks_on()
    ax1.grid(True, which='minor', linestyle=':', alpha=0.5)

    # MAE plot
    if 'mae' in data['history']:
        train_mae = data['history']['mae']
        val_mae = data['history']['val_mae']
    else:
        # If MAE isn't directly available, calculate from MSE
        train_mae = np.sqrt(data['history']['loss'])
        val_mae = np.sqrt(data['history']['val_loss'])

    ax2.plot(train_mae, label='Training MAE', color='blue')
    ax2.plot(val_mae, label='Validierung MAE', color='orange')
    ax2.set_title('Training und Validierung MAE')
    ax2.set_xlabel('Epoche')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(limits['mae']['min'], limits['mae']['max'])

    # Add minor gridlines for better readability
    ax2.minorticks_on()
    ax2.grid(True, which='minor', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mae_mse_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Standard Accuracy
    if 'accuracy' in data['history']:
        plt.figure(figsize=(10, 6))
        plt.plot(data['history']['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(data['history']['val_accuracy'], label='Validierung Accuracy', color='orange')
        plt.title('Training und Validierung Accuracy')
        plt.xlabel('Epoche')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.ylim(limits['accuracy']['min'], limits['accuracy']['max'])
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot Cumulated Accuracy
    if 'CumulatedAccuracy' in data['history']:
        plt.figure(figsize=(10, 6))
        plt.plot(data['history']['CumulatedAccuracy'], label='Training Cumulated Accuracy', color='blue')
        plt.plot(data['history']['val_CumulatedAccuracy'], label='Validierung Cumulated Accuracy', color='orange')
        plt.title('Training und Validierung Cumulated Accuracy')
        plt.xlabel('Epoche')
        plt.ylabel('Cumulated Accuracy')
        plt.legend()
        plt.grid(True)
        plt.ylim(limits['accuracy']['min'], limits['accuracy']['max'])
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'cumulated_accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    return plots_dir

def loading_animation(duration=1.5):
    animation = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    start_time = time.time()
    i = 0
    while time.time() - start_time < duration:
        print(f"\r{animation[i]} Processing...", end="")
        time.sleep(0.1)
        i = (i + 1) % len(animation)
    print("\r" + " " * 20 + "\r", end="")


# Add new menu option
def print_menu():
    print("\nChoose an action:")
    print("-" * 20)
    print("1. 📊 Calculate All Metrics")
    print("2. 📈 Plot Training Curves")
    print("3. 📉 Plot Confusion Matrices")
    print("4. 🔙 Go Back")
    print("5. 🚪 Exit")
    print("-" * 20)


def plot_confusion_matrix(folder_name):
    """
    Erstellt und speichert Standard und Cumulated Confusion Matrices als separate Dateien.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # JSON Datei einlesen
    file_path = os.path.join('evaluate', folder_name, 'results.json')
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Verzeichnis erstellen
    plots_dir = os.path.join('evaluate', folder_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Vorhersagen und wahre Labels holen
    predictions = np.array(data['predictions'])
    true_labels = np.array(data['true_labels'])
    model_type = data['model_type']

    # Konvertierung zu Klassen (1-9)
    if model_type in ['hierarchical', 'classification']:
        pred_classes = np.argmax(predictions, axis=1) + 1
        true_classes = np.argmax(true_labels, axis=1) + 1
    else:  # regression
        pred_classes = np.round(np.array([p[0] for p in predictions])).astype(int)
        true_classes = np.round(np.array([t[0] for t in true_labels])).astype(int)

    # Standard Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(true_classes, pred_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=range(1,10), yticklabels=range(1,10))
    plt.title(f'Confusion Matrix - {model_type.capitalize()} Modell\n' +
             f'Genauigkeit: {np.sum(np.diag(cm))/np.sum(cm):.4f}')
    plt.xlabel('Vorhergesagte Blühstärke')
    plt.ylabel('Tatsächliche Blühstärke')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix_standard.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Cumulated Confusion Matrix (±1 Toleranz)
    plt.figure(figsize=(12, 10))
    n_classes = 9
    cm_cumulated = np.zeros((n_classes, n_classes))

    for i in range(len(true_classes)):
        true_idx = true_classes[i] - 1
        pred_idx = pred_classes[i] - 1

        # Markiere Vorhersage als korrekt wenn innerhalb ±1
        if abs(true_idx - pred_idx) <= 1:
            cm_cumulated[true_idx, true_idx] += 1
        else:
            cm_cumulated[true_idx, pred_idx] += 1

    cm_cumulated_normalized = cm_cumulated.astype('float') / cm_cumulated.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_cumulated_normalized, annot=cm_cumulated.astype(int), fmt='d', cmap='Blues',
                xticklabels=range(1,10), yticklabels=range(1,10))
    plt.title(f'Confusion Matrix mit ±1 Toleranz - {model_type.capitalize()} Modell\n' +
             f'Kumulative Genauigkeit: {np.sum(np.diag(cm_cumulated))/np.sum(cm_cumulated):.4f}')
    plt.xlabel('Vorhergesagte Blühstärke')
    plt.ylabel('Tatsächliche Blühstärke')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix_cumulated.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Speichere numerische Matrizen
    np.savetxt(os.path.join(plots_dir, 'confusion_matrix_standard.txt'), cm, fmt='%d')
    np.savetxt(os.path.join(plots_dir, 'confusion_matrix_cumulated.txt'), cm_cumulated, fmt='%d')

    return plots_dir

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

        # Calculate cumulated accuracy accepting neighboring classes
        diff = np.abs(pred_classes - true_classes)
        cumulated_acc = np.mean(diff <= 1)

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

        # For regression, use the ±1 criterion for cumulated accuracy
        diff = np.abs(rounded_preds - true_values)
        cumulated_acc = np.mean(diff <= 1)

    # Create metrics directory and save results as before...
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
                print("\n🔄 Creating training curves...")
                loading_animation()
                try:
                    plots_dir = plot_training_curves(folder_name)
                    print_success(f"Training curves saved in {plots_dir}")
                    print("\n" + "=" * 50)
                except Exception as e:
                    print_error(f"An error occurred: {str(e)}")
            elif choice == '3':
                print("\n🔄 Creating confusion matrices...")
                loading_animation()
                try:
                    plots_dir = plot_confusion_matrix(folder_name)
                    print_success(f"Confusion matrices saved in {plots_dir}")
                    print("\n" + "=" * 50)
                except Exception as e:
                    print_error(f"An error occurred: {str(e)}")
            elif choice == '4':
                print("\n🔙 Going back to folder selection...")
                break
            elif choice == '5':
                print("\n👋 Goodbye!")
                exit(0)
            else:
                print_error("Please enter 1, 2, 3, 4, or 5")