from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from datetime import datetime

from matplotlib import pyplot as plt

from model_class.model_config import BaseModel, ModelConfig


class RegressionModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_model(self):
        for layer in self.config.pretrained_model.layers:
            layer.trainable = False
        x = self.config.pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(self.config.dense_units, activation=self.config.activation, kernel_regularizer=l2(0.001))(x)
        x = Dense(1, activation='linear')(x)

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

        history = model.fit(
            train_data[0],  # train_images
            train_data[1],  # train_labels
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint]
        )

        self.log_model_info(model, history, experiment_directory)
        return history

    def create_experiment_directory(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_directory = f"experiments/regression/regression_{self.config.pretrained_model.name}_{timestamp}"
        os.makedirs(experiment_directory, exist_ok=True)
        return experiment_directory

    def log_model_info(self, model, history, experiment_directory):
        with open(f"{experiment_directory}/model_info.txt", "w") as f:
            f.write("Regression Model Information\n")
            f.write("============================\n\n")

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

            f.write("Model Architecture:\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("\n")

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
