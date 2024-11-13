import json
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score
import time



def get_standard_limits():
    """
    Definiert Standard-Grenzen für verschiedene Metriken.

    Returns:
        dict: Dictionary mit Min/Max-Werten für jede Metrik
    """
    return {
        'mse': {
            'min': 1.0,  # 10^0
            'max': 10.0  # 10^1
        },
        'mae': {
            'min': 0.5,  # 5*10^-1
            'max': 5.0  # 5*10^0
        },
        'accuracy': {
            'min': 0.0,
            'max': 1.0
        }
    }


import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
import json


def plot_regression_scatter(folder_name):
    """
    Erstellt ein Streudiagramm der Vorhersagen vs. tatsächliche Werte
    mit perfekter Vorhersagelinie und ±2 Toleranzlinien
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # JSON Datei einlesen
    file_path = os.path.join('evaluate', folder_name, 'results.json')
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Verzeichnis erstellen
    plots_dir = os.path.join('evaluate', folder_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Daten vorbereiten
    pred = np.array([p[0] for p in data['predictions']])
    true = np.array([t[0] for t in data['true_labels']])

    # Plot erstellen mit größerer Figur
    plt.figure(figsize=(16, 12))

    # Streudiagramm
    plt.scatter(true, pred, alpha=0.5, color='blue', s=100)  # Größere Punkte

    # Dynamische Grenzen berechnen mit etwas Padding
    min_val = min(min(true), min(pred)) - 0.5
    max_val = max(max(true), max(pred)) + 0.5

    # Perfekte Vorhersage Linie und Toleranzlinien
    perfect_line = np.linspace(min_val, max_val, 100)

    # Perfekte Vorhersage (rot)
    plt.plot(perfect_line, perfect_line, 'r-', linewidth=2,
             label='Perfekte Vorhersage')

    # +2/-2 Abweichungslinien (gestrichelt)
    plt.plot(perfect_line, perfect_line + 2, 'k--', alpha=0.7, linewidth=1.5,
             label='Toleranzbereich (±2 Blühstärken)')
    plt.plot(perfect_line, perfect_line - 2, 'k--', alpha=0.7, linewidth=1.5)

    # Beschriftungen und Titel
    plt.xlabel('Tatsächliche Blühstärke', fontsize=14)
    plt.ylabel('Vorhergesagte Blühstärke', fontsize=14)
    plt.title('Streudiagramm: Vorhersagen vs. Tatsächliche Werte',
              fontsize=16, pad=20)

    # Legende
    plt.legend(fontsize=12, loc='upper left')

    # Achsengrenzen setzen
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    # Größere Achsenbeschriftungen in 1er Schritten
    plt.xticks(np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1), fontsize=12)
    plt.yticks(np.arange(np.floor(min_val), np.ceil(max_val) + 1, 1), fontsize=12)

    # Raster
    plt.grid(True, linestyle='--', alpha=0.7)

    # Layout anpassen mit mehr Platz
    plt.tight_layout()

    # Speichern in hoher Auflösung
    plt.savefig(os.path.join(plots_dir, 'regression_scatter.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    return plots_dir


def plot_experiment_comparison():
    # Vollständige Daten aus der Analyse
    data = {
        'Klassifikation': {
            'ResNet': {'acc': 38.50, 'mae': 1.46, 'mse': 5.28}, #  für acc fertig
            'MobileNet': {'acc': 39.67, 'mae': 1.07, 'mse': 2.54} #fertig
        },
        'Regression': {
            'ResNet': {'acc': 16.67, 'mae': 1.53, 'mse': 3.29}, # fertig
            'MobileNet': {'acc': 32.86, 'mae': 1.086, 'mse': 1.96} #fertig
        },
        'Hierarchical PLL': {
            'ResNet': {'acc': 38.01, 'cum_acc': 75.73, 'mae': 1.14, 'mse': 3.11},  # Updated mit echtem MSE fert
            'MobileNet': {'acc': 40.14, 'cum_acc': 80.28, 'mae': 0.9507, 'mse': 2.15}  # Beispielwerte für acc fertig
        }
    }

    experiments = list(data.keys())
    width = 0.35

    # 1. Plot Accuracy
    fig_acc, ax_acc = plt.subplots(figsize=(15, 8))
    x_acc = np.arange(len(experiments))

    # Plot standard accuracy bars
    for i, exp in enumerate(experiments):
        resnet_val = data[exp]['ResNet']['acc']
        mobilenet_val = data[exp]['MobileNet']['acc']

        # Plot standard accuracy
        resnet_bar = ax_acc.bar(x_acc[i] - width / 2, resnet_val, width,
                                color=plt.cm.Set2(0))
        mobilenet_bar = ax_acc.bar(x_acc[i] + width / 2, mobilenet_val, width,
                                   color=plt.cm.Set2(1))

        # Add labels for standard accuracy
        ax_acc.text(x_acc[i] - width / 2, resnet_val, f'ResNet\n{resnet_val:.2f}',
                    ha='center', va='bottom', fontsize=8)
        ax_acc.text(x_acc[i] + width / 2, mobilenet_val, f'MobileNet\n{mobilenet_val:.2f}',
                    ha='center', va='bottom', fontsize=8)

        # For hierarchical model, add cumulated accuracy as a lighter bar on top
        if exp == 'Hierarchical PLL':
            resnet_cum = data[exp]['ResNet']['cum_acc']
            mobilenet_cum = data[exp]['MobileNet']['cum_acc']

            # Plot cumulated accuracy with pattern
            resnet_cum_bar = ax_acc.bar(x_acc[i] - width / 2, resnet_cum - resnet_val, width,
                                        bottom=resnet_val, color=plt.cm.Set2(0), alpha=0.5)
            mobilenet_cum_bar = ax_acc.bar(x_acc[i] + width / 2, mobilenet_cum - mobilenet_val, width,
                               bottom=mobilenet_val, color=plt.cm.Set2(1), alpha=0.5)

            # Add labels for cumulated accuracy
            ax_acc.text(x_acc[i] - width / 2, resnet_cum, f'Cumulated Accuracy.: {resnet_cum:.2f}',
                        ha='center', va='bottom', fontsize=8)
            ax_acc.text(x_acc[i] + width / 2, mobilenet_cum, f'Cumulated Accuracy.: {mobilenet_cum:.2f}',
                        ha='center', va='bottom', fontsize=8)

    ax_acc.set_ylabel('Genauigkeit')
    ax_acc.set_title('Vergleich der Genauigkeit in den Basisexperimenten: ResNet vs MobileNet')
    ax_acc.set_xticks(x_acc)
    ax_acc.set_xticklabels(experiments)
    ax_acc.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('experiment_comparison_accuracy.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Plot MAE and MSE wie gehabt...
    metrics = ['mae', 'mse']
    metric_names = ['MAE', 'MSE']
    x_err = np.arange(len(experiments) * 2)

    fig_err, ax_err = plt.subplots(figsize=(15, 8))

    for i, exp in enumerate(experiments):
        for j, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            pos = i * 2 + j
            resnet_val = data[exp]['ResNet'][metric]
            mobilenet_val = data[exp]['MobileNet'][metric]

            resnet_bar = ax_err.bar(pos - width / 2, resnet_val, width,
                                    color=plt.cm.Set2(j * 2))
            mobilenet_bar = ax_err.bar(pos + width / 2, mobilenet_val, width,
                                       color=plt.cm.Set2(j * 2 + 1))

            ax_err.text(pos - width / 2, resnet_val, f'ResNet\n{resnet_val:.2f}',
                        ha='center', va='bottom', fontsize=8)
            ax_err.text(pos + width / 2, mobilenet_val, f'MobileNet\n{mobilenet_val:.2f}',
                        ha='center', va='bottom', fontsize=8)

    ax_err.set_ylabel('Fehlerwerte')
    ax_err.set_title('Vergleich der Fehlermetriken: ResNet vs MobileNet')
    ax_err.set_xticks(x_err)

    labels = []
    for exp in experiments:
        for metric_name in metric_names:
            labels.append(f'{exp}\n{metric_name}')
    ax_err.set_xticklabels(labels, rotation=0)

    ax_err.grid(True, linestyle='--', alpha=0.7)
    ax_err.set_ylim(0, 6)
    plt.tight_layout()
    plt.savefig('experiment_comparison_errors.png', bbox_inches='tight', dpi=300)
    plt.close()



def plot_training_curves(folder_name):
    """
    Plot training and validation curves with loglinear scale and standard limits
    """
    import matplotlib.pyplot as plt
    import numpy as np

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

    # Plot MSE with semilogy
    plt.figure(figsize=(10, 6))
    plt.semilogy(data['history']['loss'], label='Training MSE', color='blue')
    plt.semilogy(data['history']['val_loss'], label='Validierung MSE', color='orange')
    plt.title('Training und Validierung MSE')
    plt.xlabel('Epoche')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.ylim(limits['mse']['min'], limits['mse']['max'])
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mse_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot MAE with semilogy
    if 'mae' in data['history']:
        train_mae = data['history']['mae']
        val_mae = data['history']['val_mae']
    else:
        # If MAE isn't directly available, calculate from MSE
        train_mae = np.sqrt(data['history']['loss'])
        val_mae = np.sqrt(data['history']['val_loss'])

    plt.figure(figsize=(10, 6))
    plt.semilogy(train_mae, label='Training MAE', color='blue')
    plt.semilogy(val_mae, label='Validierung MAE', color='orange')
    plt.title('Training und Validierung MAE')
    plt.xlabel('Epoche')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    plt.ylim(limits['mae']['min'], limits['mae']['max'])
    plt.minorticks_on()
    plt.grid(True, which='minor', linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mae_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Standard Accuracy (linear scale)
    if 'accuracy' in data['history']:
        plt.figure(figsize=(10, 6))
        plt.plot(data['history']['accuracy'], label='Training Accuracy', color='blue')
        plt.plot(data['history']['val_accuracy'], label='Validierung Accuracy', color='orange')
        plt.title('Training und Validierung Accuracy')
        plt.xlabel('Epoche')
        plt.ylabel('Accuracy')
        plt.legend(['Training', 'Validierung'], loc='upper right')
        plt.grid(True)
        plt.ylim(limits['accuracy']['min'], limits['accuracy']['max'])
        plt.savefig(os.path.join(plots_dir, 'accuracy_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Plot Cumulated Accuracy (linear scale)
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

def calculate_precision_recall(folder_name):
    """
    Calculate and save precision and recall metrics for each class.
    Also calculates macro and micro averages.
    """
    import numpy as np
    from sklearn.metrics import precision_recall_fscore_support
    import os
    import json

    # Read the JSON file
    file_path = os.path.join('evaluate', folder_name, 'results.json')
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get predictions and true labels
    predictions = np.array(data['predictions'])
    true_labels = np.array(data['true_labels'])
    model_type = data['model_type']

    # Convert predictions to class labels (1-9)
    if model_type in ['hierarchical', 'classification']:
        pred_classes = np.argmax(predictions, axis=1) + 1
        true_classes = np.argmax(true_labels, axis=1) + 1
    else:  # regression
        pred_values = np.array([p[0] for p in predictions])
        true_values = np.array([t[0] for t in true_labels])
        pred_classes = np.clip(np.round(pred_values), 1, 9).astype(int)
        true_classes = np.clip(np.round(true_values), 1, 9).astype(int)

    # Calculate precision and recall for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        true_classes,
        pred_classes,
        labels=range(1, 10),  # 9 classes (1-9)
        zero_division=0
    )

    # Calculate macro and micro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_classes,
        pred_classes,
        average='macro',
        zero_division=0
    )

    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        true_classes,
        pred_classes,
        average='micro',
        zero_division=0
    )

    # Create metrics directory
    metrics_dir = os.path.join('evaluate', folder_name, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Save detailed metrics to file
    filename = os.path.join(metrics_dir, 'precision_recall_metrics.txt')
    with open(filename, 'w') as f:
        f.write("Precision and Recall Metrics\n")
        f.write("===========================\n\n")

        f.write("Per-Class Metrics:\n")
        f.write("-----------------\n")
        for i in range(9):
            f.write(f"Class {i + 1}:\n")
            f.write(f"  Precision: {precision[i]:.4f}\n")
            f.write(f"  Recall: {recall[i]:.4f}\n")
            f.write(f"  F1-Score: {f1[i]:.4f}\n")
            f.write(f"  Support: {support[i]}\n\n")

        f.write("\nAggregate Metrics:\n")
        f.write("-----------------\n")
        f.write(f"Macro Averages:\n")
        f.write(f"  Precision: {macro_precision:.4f}\n")
        f.write(f"  Recall: {macro_recall:.4f}\n")
        f.write(f"  F1-Score: {macro_f1:.4f}\n\n")

        f.write(f"Micro Averages:\n")
        f.write(f"  Precision: {micro_precision:.4f}\n")
        f.write(f"  Recall: {micro_recall:.4f}\n")
        f.write(f"  F1-Score: {micro_f1:.4f}\n\n")

        f.write("\nModel Information:\n")
        f.write(f"Model Type: {model_type}\n")
        if model_type == 'regression':
            f.write('Note: Predictions were rounded to nearest integer for metric calculation\n')
        f.write(f"Number of samples: {len(predictions)}\n")

    print(f"\nPrecision and Recall metrics saved in {filename}")
    print("\nPer-Class Summary:")
    print("-----------------")
    for i in range(9):
        print(f"Class {i + 1}: Precision = {precision[i]:.4f}, Recall = {recall[i]:.4f}")
    print("\nAggregate Metrics:")
    print("-----------------")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Micro Precision: {micro_precision:.4f}")
    print(f"Micro Recall: {micro_recall:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_avg': {
            'precision': macro_precision,
            'recall': macro_recall,
            'f1': macro_f1
        },
        'micro_avg': {
            'precision': micro_precision,
            'recall': micro_recall,
            'f1': micro_f1
        }
    }


def plot_confusion_matrix(folder_name):
    """
    Erstellt und speichert die Standard Confusion Matrix als separate Datei.
    Unterstützt sowohl Regressions- als auch Klassifikationsmodelle.
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
        # Stelle sicher, dass die Werte zwischen 1 und 9 liegen
        pred_values = np.array([p[0] for p in predictions])
        true_values = np.array([t[0] for t in true_labels])

        # Runde auf nächste ganze Zahl und beschränke auf gültige Werte
        pred_classes = np.clip(np.round(pred_values), 1, 9).astype(int)
        true_classes = np.clip(np.round(true_values), 1, 9).astype(int)

    # Standard Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(true_classes, pred_classes, labels=range(1, 10))

    # Normalisierung mit Behandlung von Null-Zeilen
    row_sums = cm.sum(axis=1)
    cm_normalized = np.zeros_like(cm, dtype=float)
    for i in range(cm.shape[0]):
        if row_sums[i] > 0:
            cm_normalized[i] = cm[i] / row_sums[i]

    sns.heatmap(cm_normalized, annot=cm, fmt='d', cmap='Blues',
                xticklabels=range(1, 10), yticklabels=range(1, 10),
                cbar_kws={'label': 'Anteil der Vorhersagen'})
    plt.xlabel('Vorhergesagte Blühstärke')
    plt.ylabel('Tatsächliche Blühstärke')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix_standard.png'), dpi=300, bbox_inches='tight')
    plt.close()

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



def print_header():
    print("\n" + "=" * 50)
    print("🔍 Bloom Strength Evaluation Tool".center(50))
    print("=" * 50 + "\n")


def print_success(text):
    print(f"\n✅ {text}")


def print_error(text):
    print(f"\n❌ {text}")


def select_folder():
    print("0: Plot Experiment Comparison")
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
            if 0 <= number <= len(folders):
                # if 0, plot experiment comparison
                if number == 0:
                    plot_experiment_comparison()
                    print_success("Experiment comparison plots saved successfully!")
                    continue
                folder_name = folders[number - 1]
                print_success(f"Selected: {folder_name}")
                return folder_name
            print_error(f"Please enter a number between 1 and {len(folders)}")
        except ValueError:
            print_error("Please enter a valid number")


def visualize_precision_recall(folder_name):
    """
    Creates comprehensive visualizations for precision and recall metrics.
    Includes:
    1. Bar chart comparing precision and recall for each class
    2. Spider/Radar plot showing precision and recall patterns
    3. Heatmap showing precision, recall, and F1 score together
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import os
    import json
    from sklearn.metrics import precision_recall_fscore_support

    # Read the JSON file
    file_path = os.path.join('evaluate', folder_name, 'results.json')
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Get predictions and true labels
    predictions = np.array(data['predictions'])
    true_labels = np.array(data['true_labels'])
    model_type = data['model_type']

    # Convert to class labels
    if model_type in ['hierarchical', 'classification']:
        pred_classes = np.argmax(predictions, axis=1) + 1
        true_classes = np.argmax(true_labels, axis=1) + 1
    else:  # regression
        pred_values = np.array([p[0] for p in predictions])
        true_values = np.array([t[0] for t in true_labels])
        pred_classes = np.clip(np.round(pred_values), 1, 9).astype(int)
        true_classes = np.clip(np.round(true_values), 1, 9).astype(int)

    # Calculate metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        true_classes,
        pred_classes,
        labels=range(1, 10),
        zero_division=0
    )

    # Create plots directory
    plots_dir = os.path.join('evaluate', folder_name, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Bar Chart
    plt.figure(figsize=(15, 8))
    x = np.arange(9)
    width = 0.35

    plt.bar(x - width / 2, precision, width, label='Precision', color='skyblue')
    plt.bar(x + width / 2, recall, width, label='Recall', color='lightcoral')

    plt.xlabel('Blühstärke')
    plt.ylabel('Score')
    plt.title('Precision und Recall pro Klasse')
    plt.xticks(x, [f'{i + 1}' for i in range(9)])
    plt.legend()

    # Add value labels on the bars
    for i, v in enumerate(precision):
        plt.text(i - width / 2, v, f'{v:.2f}', ha='center', va='bottom')
    for i, v in enumerate(recall):
        plt.text(i + width / 2, v, f'{v:.2f}', ha='center', va='bottom')

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_recall_bars.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Spider/Radar Plot
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)

    # Close the plot by appending first value
    values_precision = np.concatenate((precision, [precision[0]]))
    values_recall = np.concatenate((recall, [recall[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values_precision, 'o-', linewidth=2, label='Precision', color='skyblue')
    ax.fill(angles, values_precision, alpha=0.25, color='skyblue')
    ax.plot(angles, values_recall, 'o-', linewidth=2, label='Recall', color='lightcoral')
    ax.fill(angles, values_recall, alpha=0.25, color='lightcoral')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([f'{i + 1}' for i in range(9)])
    ax.set_title('Precision-Recall Radar Plot')
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_recall_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Heatmap
    metrics_matrix = np.array([precision, recall, f1])
    plt.figure(figsize=(15, 6))
    sns.heatmap(metrics_matrix,
                annot=True,
                fmt='.2f',
                cmap='RdYlBu_r',
                xticklabels=[f'{i + 1}' for i in range(9)],
                yticklabels=['Precision', 'Recall', 'F1'],
                cbar_kws={'label': 'Score'})

    plt.xlabel('Blühstärke')
    plt.title('Precision, Recall, und F1-Score Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'precision_recall_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return plots_dir

def print_menu():
    print("\nChoose an action:")
    print("-" * 20)
    print("1. 📊 Calculate All Metrics")
    print("2. 📈 Plot Training Curves")
    print("3. 📉 Plot Confusion Matrices")
    print("4. 📊 Plot Regression Scatter")
    print("5. 📋 Calculate Precision/Recall")  # New option
    print("6. 🔙 Go Back")
    print("7. 🚪 Exit")
    print("-" * 20)

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
                print("\n🔄 Creating regression scatter plot...")
                loading_animation()
                try:
                    plots_dir = plot_regression_scatter(folder_name)
                    print_success(f"Regression scatter plot saved in {plots_dir}")
                    print("\n" + "=" * 50)
                except Exception as e:
                    print_error(f"An error occurred: {str(e)}")
            elif choice == '5':
                    print("\n🔄 Calculating precision and recall metrics...")
                    loading_animation()
                    try:
                        calculate_precision_recall(folder_name)
                        print_success("Precision and recall metrics calculated and saved successfully!")
                        print("\n" + "=" * 50)
                    except Exception as e:
                        print_error(f"An error occurred: {str(e)}")
            elif choice == '6':
                        print("\n🔄 Creating precision and recall visualizations...")
                        loading_animation()
                        try:
                            plots_dir = visualize_precision_recall(folder_name)
                            print_success(f"Precision and recall visualizations saved in {plots_dir}")
                            print("\n" + "=" * 50)
                        except Exception as e:
                            print_error(f"An error occurred: {str(e)}")
            elif choice == '7':
                    print("\n🔙 Going back to folder selection...")
                    break
            elif choice == '8':
                print("\n👋 Goodbye!")
                exit(0)
            else:
                print_error("Please enter a number between 1 and 8")
