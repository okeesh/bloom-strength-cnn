from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import seaborn as sns


class BaseModel(ABC):
    @abstractmethod
    def create_model(self):
        pass

    @abstractmethod
    def compile_model(self, model):
        pass

    @abstractmethod
    def train_model(self, model, train_data, validation_data):
        pass

    def plot_confusion_matrix(self, y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(filename)
        plt.close()


class NormalizedRegressionModel(BaseModel):
    def __init__(self, config):
        self.config = config
        self.model = self.create_model()

    def create_model(self):
        base_model = self.config.pretrained_model
        for layer in base_model.layers:
            layer.trainable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(self.config.dense_units, activation='relu', kernel_regularizer=l2(self.config.regularization_rate))(x)
        x = Dense(1, activation='linear')(x)  # Linear activation for regression

        model = Model(inputs=base_model.input, outputs=x)
        return model

    def compile_model(self, model):
        model.compile(optimizer=Adam(learning_rate=self.config.learning_rate),
                      loss='mse',
                      metrics=['mae'])
        return model

    @staticmethod
    def normalize_labels(labels):
        return (labels - 1) / 8

    @staticmethod
    def rescale_predictions(normalized_predictions):
        return normalized_predictions * 8 + 1

    def train_model(self, model, train_data, validation_data):
        train_images, train_labels = train_data
        validation_images, validation_labels = validation_data

        train_labels_normalized = self.normalize_labels(train_labels)
        validation_labels_normalized = self.normalize_labels(validation_labels)

        history = model.fit(
            train_images,
            train_labels_normalized,
            validation_data=(validation_images, validation_labels_normalized),
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=1
        )

        # Predict on validation data and plot confusion matrix
        validation_predictions = self.predict(validation_images)
        rounded_predictions = np.round(validation_predictions).astype(int)
        rounded_labels = np.round(validation_labels).astype(int)
        classes = list(range(1, 10))  # Assuming classes are 1 to 9
        self.plot_confusion_matrix(rounded_labels, rounded_predictions, classes, 'regression_confusion_matrix.png')

        return history

    def predict(self, images):
        normalized_predictions = self.model.predict(images)
        return self.rescale_predictions(normalized_predictions)

    def evaluate(self, validation_images, validation_labels):
        predictions = self.predict(validation_images)
        mse = np.mean((predictions - validation_labels) ** 2)
        mae = np.mean(np.abs(predictions - validation_labels))
        return {'mse': mse, 'mae': mae}

    def plot_predictions(self, true_labels, predicted_labels):
        plt.figure(figsize=(10, 6))
        plt.scatter(true_labels, predicted_labels, alpha=0.5)
        plt.plot([1, 9], [1, 9], 'r--')  # Diagonal line for perfect predictions
        plt.xlabel("True Labels")
        plt.ylabel("Predicted Labels")
        plt.title("Predicted vs True Labels")
        plt.show()