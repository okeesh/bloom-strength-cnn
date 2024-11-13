import tensorflow as tf

tf.config.run_functions_eagerly(True)
from keras.applications import MobileNet, ResNet50, MobileNetV2
from dataset.load_data import load_data
from model_class.model import ModelTrainer
from model_class.model_config import ModelConfig
import time
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt

# Define input shape
input_shape = (224, 224, 3)


def train_classification_model():
    print("Training classification model...")
    config_classification = ModelConfig(
        model_type='classification',
        learning_rate=0.001,
        dropout_rate=0.1,
        regularization_rate=0.001,
        batch_size=16,
        epochs=30,
        pretrained_model=MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
        dense_units=256,
        num_classes=9,
        input_shape=input_shape
    )
    model = ModelTrainer(config_classification)

    # Load train and validation data from load_data
    train_images, train_labels, validation_images, validation_labels, class_weights = load_data(
        model_type='classification')

    # Use the resampled data for training
    train_data = (train_images, train_labels, class_weights)
    validation_data = (validation_images, validation_labels)

    print("Loaded Class Weights ", class_weights)

    # Train the model
    history = model.train(train_data, validation_data)
    print(history.history)


def gs_classification_model():
    print("Training classification model with grid search...")

    # Define the parameter grid
    param_grid = {
        'dropout_rate': [0.1, 0.3, 0.5],
        'batch_size': [16, 32, 64],
        'dense_units': [256, 512, 1024]
    }

    # Generate all combinations of parameters
    grid = list(ParameterGrid(param_grid))
    total_iterations = len(grid)
    print(f"Total number of iterations: {total_iterations}")

    best_val_accuracy = 0
    best_params = None
    best_history = None
    all_results = []

    # Load train and validation data
    train_images, train_labels, validation_images, validation_labels, class_weights = load_data(
        model_type='classification')

    train_data = (train_images, train_labels)
    validation_data = (validation_images, validation_labels)

    start_time = time.time()

    for i, params in enumerate(grid, 1):
        print(f"\nIteration {i}/{total_iterations}")
        print(f"Training with parameters: {params}")

        config_classification = ModelConfig(
            model_type='classification',
            learning_rate=0.001,
            activation="softmax",
            regularization_rate=0.001,
            dropout_rate=params['dropout_rate'],
            batch_size=params['batch_size'],
            epochs=30,
            pretrained_model=MobileNetV2(weights=None, include_top=False, input_shape=input_shape),
            dense_units=params['dense_units'],
            num_classes=9,
            input_shape=input_shape
        )

        model = ModelTrainer(config_classification)

        # Train the model
        history = model.train(train_data, validation_data)

        # Get the best validation accuracy for this configuration
        val_accuracy = max(history.history['val_accuracy'])

        # Store the results
        result = {**params, 'val_accuracy': val_accuracy}
        all_results.append(result)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = params
            best_history = history

        # Calculate and print progress
        elapsed_time = time.time() - start_time
        avg_time_per_iteration = elapsed_time / i
        estimated_time_remaining = avg_time_per_iteration * (total_iterations - i)

        print(f"Current best validation accuracy: {best_val_accuracy:.4f}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")
        print(f"Estimated completion time: {time.ctime(time.time() + estimated_time_remaining)}")

    print("\nGrid search completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best validation accuracy: {best_val_accuracy}")

    # Plot training history for the best model
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(best_history.history['loss'], label='Training Loss')
    plt.plot(best_history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(best_history.history['accuracy'], label='Training Accuracy')
    plt.plot(best_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot sample images
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(train_images[i])
        ax.axis('off')
        ax.set_title(f"Label: {train_labels[i]}")
    plt.tight_layout()
    plt.show()

    return all_results, best_params, best_val_accuracy

# dropout ist beim training aktiv, beim training deutlich kürzeres netz als beim testen
# dadurch geringere accuracy
# auf validierungsdaten wird das ganze nett genutzt
# deshalb val daten besser

#mobile net liefert stabilere Ergebnisse als resnet bei hierarchical, aber nicht besser
# trainingsdaten besser muss nichts heißen, kann auch wegen dropout oder netzstruktur könnte auf overfitting andeuten
#Overfitting zu sehen





def train_regression_model():
    print("Training regression model...")
    config_regression = ModelConfig(
        model_type='regression',
        learning_rate=0.001,
        activation="linear",
        regularization_rate=0.001,
        dropout_rate=0.1,
        batch_size=16,
        epochs=30,
        pretrained_model=MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
        dense_units=256,
        num_classes=1,
        input_shape=input_shape
    )

    model = ModelTrainer(config_regression)

    # Load train and validation data from load_data
    train_images, train_labels, validation_images, validation_labels, class_weights = load_data(model_type='regression')

    train_data = (train_images, train_labels)
    validation_data = (validation_images, validation_labels)


    # Train the model
    history = model.train(train_data, validation_data)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(train_images[i])
        ax.axis('off')
        ax.set_title(f"Label: {train_labels[i]}")
    plt.tight_layout()
    plt.show()

    # Optionally, you can print or plot the training history
    print(history.history)


def train_hierarchical_model():
    print("Training hierarchical model...")
    # Configuration for Hierarchical Model
    config_hierarchical = ModelConfig(
        model_type='hierarchical',
        learning_rate=0.001,
        dropout_rate=0.1,
        regularization_rate=0.001,
        batch_size=16,
        epochs=30,
        pretrained_model=MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape),
        dense_units=256,
        num_classes=9,
        input_shape=input_shape
    )
    model = ModelTrainer(config_hierarchical)
    # Load train and validation data from load_data
    train_images, train_labels, validation_images, validation_labels, class_weights = load_data(
        model_type='hierarchical')
    train_data = (train_images, train_labels, class_weights)
    validation_data = (validation_images, validation_labels)
    # Train the model
    history = model.train(train_data, validation_data)
    print(history.history)


def train_regression_normalized_model():
    config_regression = ModelConfig(
        model_type='regression_normalized',
        learning_rate=0.001,
        activation="linear",
        regularization_rate=0.001,
        # dropout_rate=0.1,
        batch_size=16,
        epochs=50,
        pretrained_model=MobileNet(weights='imagenet', include_top=False, input_shape=input_shape),
        dense_units=256,
        input_shape=input_shape
    )

    model = ModelTrainer(config_regression)

    # Load train and validation data from load_data
    train_images, train_labels, validation_images, validation_labels, class_weights = load_data(model_type='regression')

    train_data = (train_images, train_labels)
    validation_data = (validation_images, validation_labels)

    history = model.train(train_data, validation_data)

    print(history.history)


def main():
    while True:
        print("\nChoose an option:")
        print("1. Train Classification Model")
        print("2. Train Regression Model")
        print("3. Train Hierarchical Model")
        print("4. Train Normalized Regression Model")
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")

        if choice == '1':
            train_classification_model()
        elif choice == '2':
            train_regression_model()
        elif choice == '3':
            train_hierarchical_model()
        elif choice == '4':
            train_regression_normalized_model()
        elif choice == '5':
            print("Exiting program.")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()


