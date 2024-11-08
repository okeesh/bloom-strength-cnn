import numpy as np
from keras.losses import mean_absolute_error, mean_squared_error
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
import os
from datetime import datetime
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras import backend as K
from model_class.model_config import BaseModel, ModelConfig, LearningRateScheduler
import tensorflow as tf

from model_class.utils import balanced_accuracy, BalancedDataGenerator


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


def weighted_mse(weights=None):
    def loss(y_true, y_pred):
        error = y_pred - y_true
        squared_error = tf.square(error)

        if weights is not None:
            # Convert weights dictionary to a list
            weight_list = [weights.get(i, 1.0) for i in range(max(weights.keys()) + 1)]
            weight_tensor = tf.constant(weight_list, dtype=tf.float32)

            # Apply weights based on the absolute error
            abs_error = tf.abs(error)
            abs_error_int = tf.cast(tf.clip_by_value(abs_error, 0, len(weight_list) - 1), tf.int32)
            weight = tf.gather(weight_tensor, abs_error_int)

            squared_error = squared_error * weight

        return tf.reduce_mean(squared_error)

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
        x = Dense(self.config.dense_units, activation='relu', kernel_regularizer=l2(self.config.regularization_rate))(x)
        x = Dense(1, activation=self.config.activation)(x)

        model = Model(inputs=self.config.pretrained_model.input, outputs=x)
        return model

    def compile_model(self, model, class_weights=None):
        model.compile(optimizer=Adam(lr=self.config.learning_rate),
                      loss='mse',
                      #loss=tolerant_mse(1),
                      metrics=['mae', 'mse'])
        return model

    def train_model(self, model, train_data, validation_data):
        experiment_directory = self.create_experiment_directory()
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )

        bgen = BalancedDataGenerator(train_data[0], train_data[1], datagen, batch_size=16)
        steps_per_epoch = bgen.steps_per_epoch

        y_gen = [bgen.__getitem__(0)[1] for i in range(steps_per_epoch)]
        print(np.unique(y_gen, return_counts=True))

        #Callbacks
        lr_scheduler = LearningRateScheduler(new_lr=self.config.learning_rate * 0.1, change_epoch=20)
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        # history = model.fit(
        #     datagen.flow(train_data[0], train_data[1], batch_size=self.config.batch_size),
        #     validation_data=validation_data,
        #     epochs=self.config.epochs,
        #     batch_size=self.config.batch_size,
        #     verbose=1,
        #     callbacks=[lr_scheduler, early_stopping]
        # )

        history = model.fit(
            bgen,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data,
            epochs=self.config.epochs,
            verbose=1,
            callbacks=[lr_scheduler, early_stopping]
        )




        # After training
        y_pred = model.predict(validation_data[0]).flatten()
        y_true = validation_data[1].flatten()

        # Round predictions and true values, and clip to range [1, 9]
        y_pred_rounded = np.clip(np.round(y_pred), 1, 9).astype(int)
        y_true_rounded = np.clip(np.round(y_true), 1, 9).astype(int)

        # Calculate pseudo-balanced accuracy
        bal_acc = balanced_accuracy(y_true_rounded, y_pred_rounded)
        print(f"Pseudo-Balanced Accuracy: {bal_acc:.4f}")

        # Add balanced accuracy to the history object
        history.history['balanced_accuracy'] = [bal_acc]

        predictions = model.predict(validation_data[0])
        true_labels = validation_data[1]

        self.save_results(model, history, predictions, true_labels, experiment_directory)
        self.log_model_info(model, history, experiment_directory)
        self.evaluate_regression_model(model, validation_data, experiment_directory)
        print("Confusion matrix for rounded regression predictions has been saved.")

        self.log_model_info(model, history, experiment_directory)
        return history

    def evaluate_regression_model(self, model, validation_data, experiment_directory):
        y_pred = model.predict(validation_data[0]).flatten()
        y_true = validation_data[1].flatten()

        # Cap predictions to [1, 9] range
        y_pred = np.clip(y_pred, 1, 9)

        # Calculate regression metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")

        # Scatter plot of predicted vs true values
        plt.figure(figsize=(10, 10))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([1, 9], [1, 9], 'r--')  # Diagonal line
        plt.xlabel('Wahre Werte')
        plt.ylabel('Vorhersagen')
        plt.title('Vorhersage vs Wahre Werte')
        plt.savefig(f'{experiment_directory}/vorhersage_vs_wahr_streudiagramm.png')
        plt.close()

        # Round predictions and true values, and clip to range [1, 9]
        y_pred_rounded = np.clip(np.round(y_pred), 1, 9).astype(int)
        y_true_rounded = np.clip(np.round(y_true), 1, 9).astype(int)

        # Plot confusion matrix
        self.plot_confusion_matrix(y_true_rounded, y_pred_rounded, list(range(1, 10)),
                                   f'{experiment_directory}/regression_confusion_matrix.png')

        print("Additional evaluation plots have been saved.")

    def plot_confusion_matrix(self, y_true, y_pred, classes, filename):
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title('Konfusionsmatrix')
        plt.ylabel('Wahres Label')
        plt.xlabel('Vorhergesagtes Label')
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
            f.write(f"{self.config.pretrained_model.name}\n\n")

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
            f.write(f"Pseudo-Balanced Accuracy: {history.history['balanced_accuracy'][0]:.4f}\n")

        # Create a single figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        epochs = range(1, len(history.history['mse']) + 1)

        # Plot training and validation MSE
        ax1.plot(epochs, history.history['mse'], label='Training MSE')
        ax1.plot(epochs, history.history['val_mse'], label='Validierung MSE')
        ax1.set_title('Training und Validierung MSE')
        ax1.set_ylabel('MSE')
        ax1.set_xlabel('Epoche')
        ax1.legend(loc='upper right')
        ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Plot training and validation MAE
        ax2.plot(epochs, history.history['mae'], label='Training MAE')
        ax2.plot(epochs, history.history['val_mae'], label='Validierung MAE')
        ax2.set_title('Training und Validierung MAE')
        ax2.set_ylabel('MAE')
        ax2.set_xlabel('Epoche')
        ax2.legend(loc='upper right')
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # Adjust layout and save the figure
        plt.tight_layout()
        plt.savefig(f"{experiment_directory}/training_validation_metrics.png")
        plt.close()  # Close the figure instead of showing it