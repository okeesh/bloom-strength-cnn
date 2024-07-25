import os
from datetime import datetime

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from matplotlib import pyplot as plt
from model_class.model_config import ModelConfig, BaseModel


class ClassificationModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_model(self):
        for layer in self.config.pretrained_model.layers:
            layer.trainable = False

        x = self.config.pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(self.config.dense_units, activation=self.config.activation, kernel_regularizer=l2(0.001))(x)
        x = Dense(self.config.num_classes, activation='relu')(x)

        model = Model(inputs=self.config.pretrained_model.input, outputs=x)
        return model

    def compile_model(self, model):
        model.compile(optimizer=Adam(lr=self.config.learning_rate),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def train_model(self, model, train_data, validation_data):
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = ModelCheckpoint(filepath='best_classification_model.h5', monitor='val_loss',
                                           save_best_only=True)

        history = model.fit(
            train_data[0],  # train_images (now containing only the cropped regions)
            train_data[1],  # train_labels
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint],
            class_weight=train_data[2] if len(train_data) > 2 else None  # class_weights if provided
        )

        # Give a final print for the training
        print(f"Final training accuracy: {history.history['accuracy'][-1]}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]}")

        # Print the early_stopping result if early stopped
        if early_stopping.stopped_epoch > 0:
            print(f"Early stopped at epoch {early_stopping.stopped_epoch} with validation loss {early_stopping.best}")

        return history

    def create_experiment_directory(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_directory = f"experiments/classification/classification_{self.config.pretrained_model.name}_{timestamp}"
        os.makedirs(experiment_directory, exist_ok=True)
        return experiment_directory

    def log_model_info(self, model, history, experiment_directory):
        with open(f"{experiment_directory}/model_info.txt", "w") as f:
            f.write("Classification Model Information\n")
            f.write("=================================\n\n")

            f.write("Hyperparameters:\n")
            f.write(f"Learning rate: {self.config.learning_rate}\n")
            f.write(f"Dropout rate: {self.config.dropout_rate}\n")
            f.write(f"Batch size: {self.config.batch_size}\n")
            f.write(f"Epochs: {self.config.epochs}\n")
            f.write(f"Activation: {self.config.activation}\n")
            f.write(f"Number of classes: {self.config.num_classes}\n\n")

            f.write("Model Architecture:\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write("\n")

            f.write("Training Performance:\n")
            f.write(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}\n")
            f.write(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}\n")

            # Plot training accuracy
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['accuracy'])
            plt.title('Training Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train'], loc='upper left')
            plt.savefig(f"{experiment_directory}/training_accuracy.png")
            plt.show()

            # Plot training loss
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'])
            plt.title('Training Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train'], loc='upper left')
            plt.savefig(f"{experiment_directory}/training_loss.png")
            plt.show()

