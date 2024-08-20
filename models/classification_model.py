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
        focal_loss = CategoricalFocalLoss(alpha=0.5, gamma=5.0)
        model.compile(optimizer=Adam(lr=self.config.learning_rate),
                      loss=focal_loss,
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

import tensorflow as tf
from keras import backend as K
from keras.losses import Loss

class CategoricalFocalLoss(Loss):
    def __init__(self, alpha=0.25, gamma=2.0, class_weights=None, from_logits=False, label_smoothing=0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        # Apply label smoothing if specified
        y_true = y_true * (1.0 - self.label_smoothing) + (self.label_smoothing / y_true.shape[-1])

        # If predictions are logits, apply softmax
        if self.from_logits:
            y_pred = tf.nn.softmax(y_pred, axis=-1)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate focal loss
        loss = self.alpha * K.pow(1 - y_pred, self.gamma) * cross_entropy

        # Apply class weights if specified
        if self.class_weights is not None:
            loss = loss * tf.gather(self.class_weights, K.argmax(y_true, axis=-1))

        # Sum over classes
        loss = K.sum(loss, axis=-1)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "class_weights": self.class_weights,
            "from_logits": self.from_logits,
            "label_smoothing": self.label_smoothing
        })
        return config