import numpy as np
from sklearn.metrics import confusion_matrix




def balanced_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc_per_class = cm.diagonal()
    return np.mean(acc_per_class)

import json
from sklearn.metrics import confusion_matrix

def balanced_accuracy_from_file(results_file):
    # Load the results from the file
    with open(results_file, 'r') as f:
        results = json.load(f)

    # Extract the predictions and true labels
    y_pred = np.array(results['predictions'])
    y_true = np.array(results['true_labels'])

    # Calculate the balanced accuracy
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc_per_class = cm.diagonal()
    return np.mean(acc_per_class)

from keras.utils.data_utils import Sequence
from imblearn.over_sampling import RandomOverSampler
from imblearn.keras import balanced_batch_generator


class BalancedDataGenerator(Sequence):
    """ImageDataGenerator + RandomOversampling for both classification and regression"""

    def __init__(self, x, y, datagen, batch_size=32):
        self.datagen = datagen
        self.batch_size = min(batch_size, x.shape[0])
        datagen.fit(x)

        print("Input shapes:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")

        self.is_classification = len(y.shape) > 1 and y.shape[1] > 1

        if self.is_classification:
            print("Detected classification task (one-hot encoded labels)")
            self.n_classes = y.shape[1]

            # Print original class distribution
            original_class_counts = np.sum(y, axis=0)
            print("\nOriginal class distribution:")
            for i, count in enumerate(original_class_counts):
                print(f"Class {i}: {count}")
        else:
            print("Detected regression task")
            y = y.reshape(-1, 1)
            print("\nOriginal regression target statistics:")
            print(f"Mean: {np.mean(y):.2f}")
            print(f"Std: {np.std(y):.2f}")
            print(f"Min: {np.min(y):.2f}")
            print(f"Max: {np.max(y):.2f}")

        self.gen, self.steps_per_epoch = balanced_batch_generator(
            x.reshape(x.shape[0], -1), y,
            sampler=RandomOverSampler(),
            batch_size=self.batch_size,
            keep_sparse=True
        )
        self._shape = (self.steps_per_epoch * batch_size, *x.shape[1:])

        # Sample all oversampled data to show distribution
        oversampled_y = []
        for _ in range(self.steps_per_epoch):
            _, batch_y = next(self.gen)
            oversampled_y.extend(batch_y)
        oversampled_y = np.array(oversampled_y)

        if self.is_classification:
            oversampled_class_counts = np.sum(oversampled_y, axis=0)
            print("\nOversampled class distribution:")
            for i, count in enumerate(oversampled_class_counts):
                print(f"Class {i}: {count}")
        else:
            print("\nOversampled regression target statistics:")
            print(f"Mean: {np.mean(oversampled_y):.2f}")
            print(f"Std: {np.std(oversampled_y):.2f}")
            print(f"Min: {np.min(oversampled_y):.2f}")
            print(f"Max: {np.max(oversampled_y):.2f}")

        # Reset the generator
        self.gen, _ = balanced_batch_generator(
            x.reshape(x.shape[0], -1), y,
            sampler=RandomOverSampler(),
            batch_size=self.batch_size,
            keep_sparse=True
        )

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = next(self.gen)
        x_batch = x_batch.reshape(-1, *self._shape[1:])

        if not self.is_classification:
            y_batch = y_batch.ravel()

        x_batch, y_batch = self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()
        return x_batch, y_batch


# class BalancedDataGeneratorMultiHot(Sequence):
#     """BalancedDataGenerator for hierarchical multi-class labels"""
#
#     def __init__(self, x, y, datagen, batch_size=32):
#         self.datagen = datagen
#         self.batch_size = min(batch_size, x.shape[0])
#         datagen.fit(x)
#
#         self.x = x
#         self.y = y
#
#         print("Input shapes:")
#         print(f"x shape: {x.shape}")
#         print(f"y shape: {y.shape}")
#         print(f"Sample y data:\n{y[:5]}")
#
#         self.n_classes = y.shape[1]
#
#         # Count samples per class (considering only '2' as class presence)
#         class_counts = np.sum(y == 2, axis=0)
#         print(f"Samples per class (based on '2's): {class_counts}")
#
#         # Create indices for oversampling
#         self.indices = np.arange(len(x))
#         max_count = int(np.max(class_counts))
#         self.oversampled_indices = []
#
#         for i in range(self.n_classes):
#             class_indices = self.indices[y[:, i] == 2]
#             if len(class_indices) > 0:
#                 oversampled = np.random.choice(class_indices, size=max_count, replace=True)
#                 self.oversampled_indices.extend(oversampled)
#
#         np.random.shuffle(self.oversampled_indices)
#         self.steps_per_epoch = len(self.oversampled_indices) // self.batch_size
#
#         # Print some oversampled labels
#         self.print_oversampled_labels()
#
#     def __len__(self):
#         return self.steps_per_epoch
#
#     def __getitem__(self, idx):
#         batch_indices = self.oversampled_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
#         x_batch = self.x[batch_indices]
#         y_batch = self.y[batch_indices]
#
#         x_batch, y_batch = self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()
#
#         return x_batch, y_batch
#
#     def on_epoch_end(self):
#         np.random.shuffle(self.oversampled_indices)
#
#     def print_oversampled_labels(self):
#         print("\nSample of oversampled labels:")
#         sample_size = min(20, len(self.oversampled_indices))
#         sample_indices = np.random.choice(self.oversampled_indices, size=sample_size, replace=False)
#         sample_labels = self.y[sample_indices]
#
#         for label in sample_labels:
#             print(label)
#
#         print("\nClass distribution in oversampled data (based on '2's):")
#         oversampled_class_counts = np.sum(self.y[self.oversampled_indices] == 2, axis=0)
#         print(oversampled_class_counts)
#
#         print("\nTotal samples in oversampled data:", len(self.oversampled_indices))
#
#     def get_true_labels(self):
#         """Returns the index of '2' in each label as the true class"""
#         return np.argmax(self.y == 2, axis=1)

import numpy as np
from tensorflow.keras.utils import Sequence
from collections import Counter


class BalancedDataGeneratorMultiHot(Sequence):
    """BalancedDataGenerator for hierarchical multi-class labels"""

    def __init__(self, x, y, datagen, batch_size=16):
        self.datagen = datagen
        self.batch_size = min(batch_size, x.shape[0])
        datagen.fit(x)

        self.x = x
        self.y = y

        print("Input shapes:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"Sample y data:\n{y[:5]}")

        self.n_classes = y.shape[1]

        # Get true labels (index of '2' in each row)
        self.true_labels = np.argmax(y == 2, axis=1)

        # Count samples per class
        class_counts = Counter(self.true_labels)
        print(f"Samples per class: {class_counts}")

        # Determine the desired count for each class (oversample to match the most common class)
        max_count = max(class_counts.values())

        # Create indices for oversampling
        self.oversampled_indices = []
        for class_label in range(self.n_classes):
            class_indices = np.where(self.true_labels == class_label)[0]
            if len(class_indices) > 0:
                oversampled = np.random.choice(class_indices, size=max_count, replace=True)
                self.oversampled_indices.extend(oversampled)

        np.random.shuffle(self.oversampled_indices)
        self.steps_per_epoch = len(self.oversampled_indices) // self.batch_size

        # Print oversampled distribution
        self.print_oversampled_labels()

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        batch_indices = self.oversampled_indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        x_batch = self.x[batch_indices]
        y_batch = self.y[batch_indices]

        x_batch, y_batch = self.datagen.flow(x_batch, y_batch, batch_size=self.batch_size).next()

        return x_batch, y_batch

    def on_epoch_end(self):
        np.random.shuffle(self.oversampled_indices)

    def print_oversampled_labels(self):
        oversampled_true_labels = self.true_labels[self.oversampled_indices]

        print("\nClass distribution in oversampled data:")
        unique, counts = np.unique(oversampled_true_labels, return_counts=True)
        for class_label, count in zip(unique, counts):
            print(f"Class {class_label}: {count}")

        print("\nTotal samples in oversampled data:", len(self.oversampled_indices))

        print("\nUnique classes and their counts:")
        print(np.unique(oversampled_true_labels, return_counts=True))

    def get_true_labels(self):
        """Returns the true labels for the oversampled data"""
        return self.true_labels[self.oversampled_indices]


import json
from sklearn.metrics import balanced_accuracy_score


def calculate_balanced_accuracy(file_content):
    # Parse the JSON content
    data = json.loads(file_content)

    # Extract predictions and true labels
    predictions = np.array(data['predictions'])
    true_labels = np.array(data['true_labels'])

    # Convert predictions to class labels
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = np.argmax(true_labels, axis=1)

    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

    return balanced_acc


def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Find the JSON content within the file
    start = content.find('{')
    end = content.rfind('}') + 1
    json_content = content[start:end]

    balanced_acc = calculate_balanced_accuracy(json_content)
    print(f"Balanced Accuracy: {balanced_acc:.4f}")


def plot_hierarchical_confusion_matrix(y_true, y_pred, classes, filename,
                                       highlight_acceptable=True):
    """
    Plot confusion matrix with special highlighting for hierarchical model analysis.

    Parameters:
    -----------
    y_true : array-like
        True labels (ground truth)
    y_pred : array-like
        Predicted labels
    classes : list
        List of class labels
    filename : str
        Output filename for the plot
    highlight_acceptable : bool
        If True, highlights acceptable predictions (±1) differently
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create mask for acceptable predictions (±1)
    n_classes = len(classes)
    acceptable_mask = np.zeros((n_classes, n_classes))
    for i in range(n_classes):
        for j in range(n_classes):
            if abs(i - j) <= 1:
                acceptable_mask[i, j] = 1

    # Plot settings
    plt.figure(figsize=(12, 10))

    # Plot base confusion matrix
    if highlight_acceptable:
        # Use different color for acceptable vs unacceptable predictions
        unacceptable = np.ma.masked_where(acceptable_mask == 1, cm)
        acceptable = np.ma.masked_where(acceptable_mask == 0, cm)

        # Plot unacceptable predictions in red
        sns.heatmap(unacceptable, annot=cm, fmt='d', cmap='Reds',
                    xticklabels=classes, yticklabels=classes, alpha=0.7)
        # Plot acceptable predictions in blues
        sns.heatmap(acceptable, annot=cm, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)
    else:
        # Standard confusion matrix plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=classes, yticklabels=classes)

    # Add accuracy metrics
    exact_accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    acceptable_accuracy = np.sum(cm * acceptable_mask) / np.sum(cm)

    plt.title(f'Hierarchical Confusion Matrix\n' +
              f'Exact Accuracy: {exact_accuracy:.2%}\n' +
              f'Acceptable Accuracy (±1): {acceptable_accuracy:.2%}')
    plt.ylabel('Wahres Label')
    plt.xlabel('Vorhergesagtes Label')

    # Add legend
    if highlight_acceptable:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', alpha=0.5, label='Akzeptable Abweichung (±1)'),
            Patch(facecolor='red', alpha=0.5, label='Unakzeptable Abweichung (>±1)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    # Usage
    process_file('balancedaccuracy/results.json')

