import numpy as np
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
from datetime import datetime

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from model_class.model_config import BaseModel, ModelConfig
import tensorflow as tf


def tolerant_loss(tolerance=1.0):
    def loss(y_true, y_pred):
        error = tf.abs(y_pred - y_true)
        return tf.reduce_mean(tf.maximum(error - tolerance, 0.0))

    return loss

def tolerant_mse(tolerance=1.0):
    def loss(y_true, y_pred):
        error = tf.abs(y_pred - y_true)
        squared_loss = tf.square(tf.maximum(error - tolerance, 0.0))
        return tf.reduce_mean(squared_loss)
    return loss


class RegressionModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config



    def create_model(self):
        for layer in self.config.pretrained_model.layers:
            layer.trainable = False

        x = self.config.pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(self.config.dense_units, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dense(1, activation=self.config.activation)(x)

        model = Model(inputs=self.config.pretrained_model.input, outputs=x)
        return model

    def compile_model(self, model):
        model.compile(optimizer=Adam(lr=self.config.learning_rate),
                      loss='mse',
                      metrics=['mae', 'mse'])
        return model

    def train_model(self, model, train_data, validation_data):
        experiment_directory = self.create_experiment_directory()

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = ModelCheckpoint(filepath=f'{experiment_directory}/best_regression_model.h5',
                                           monitor='val_mse',
                                           save_best_only=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        history = model.fit(
            train_data[0],
            train_data[1],
            validation_data=validation_data,
            epochs=self.config.epochs,  # Increase epochs
            batch_size=self.config.batch_size,  # Adjust batch size
            verbose=1,
            callbacks=[early_stopping, model_checkpoint, reduce_lr]
        )
        # Plot confusion matrix for rounded predictions
        y_pred = model.predict(validation_data[0]).flatten()
        y_true = validation_data[1].flatten()

        # Round predictions and true values, and clip to range [1, 9]
        y_pred_rounded = np.clip(np.round(y_pred), 1, 9).astype(int)
        y_true_rounded = np.clip(np.round(y_true), 1, 9).astype(int)

        # Explicitly define classes
        classes = list(range(1, 10))  # Classes from 1 to 9

        # Plot confusion matrix
        self.plot_confusion_matrix(y_true_rounded, y_pred_rounded, classes, f'{experiment_directory}/regression_confusion_matrix.png')

        print("Confusion matrix for rounded regression predictions has been saved.")

        self.log_model_info(model, history, experiment_directory)
        return history

    def plot_confusion_matrix(self, y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(filename)
        plt.close()

    def create_experiment_directory(self):
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        experiment_directory = f"experiments/regression/regression_{timestamp}"
        os.makedirs(experiment_directory, exist_ok=True)
        return experiment_directory

    def log_model_info(self, model, history, experiment_directory):
        with open(f"{experiment_directory}/model_info.txt", "w") as f:
            f.write("Regression Model Information\n")
            f.write("============================\n\n")

            f.write("Pretrained Model:\n")
            f.write(self.config.pretrained_model.name)

            f.write("Hyperparameters:\n")
            f.write(f"Learning rate: {self.config.learning_rate}\n")
            f.write(f"Dropout rate: {self.config.dropout_rate}\n")
            f.write(f"Batch size: {self.config.batch_size}\n")
            f.write(f"Epochs: {self.config.epochs}\n")
            f.write(f"Activation: {self.config.activation}\n\n")

            f.write("Training Performance:\n")
            f.write(f"Final training MSE: {history.history['mse'][-1]:.4f}\n")
            f.write(f"Final training MAE: {history.history['mae'][-1]:.4f}\n")
            f.write(f"Final validation MSE: {history.history['val_mse'][-1]:.4f}\n")
            f.write(f"Final validation MAE: {history.history['val_mae'][-1]:.4f}\n")

            # Plot training and validation MSE
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['mse'])
            plt.plot(history.history['val_mse'])
            plt.title('Training and Validation MSE')
            plt.ylabel('MSE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(f"{experiment_directory}/training_validation_mse.png")
            plt.show()

            # Plot training and validation MAE
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['mae'])
            plt.plot(history.history['val_mae'])
            plt.title('Training and Validation MAE')
            plt.ylabel('MAE')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            plt.savefig(f"{experiment_directory}/training_validation_mae.png")
            plt.show()
