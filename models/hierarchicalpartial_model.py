import os
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
from tensorflow import clip_by_value
from tensorflow.python.framework.indexed_slices import math_ops
from tensorflow.python.ops import clip_ops
from model_class.model_config import BaseModel, ModelConfig


class HierarchicalPartialLossModel(BaseModel):
    def __init__(self, config: ModelConfig):
        self.config = config

    def create_model(self):
        for layer in self.config.pretrained_model.layers:
            layer.trainable = False
        x = self.config.pretrained_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        x = Dense(self.config.dense_units, activation="relu", kernel_regularizer='l2')(x)
        x = Dense(self.config.num_classes, activation="softmax")(x)

        model = Model(inputs=self.config.pretrained_model.input, outputs=x)
        return model

    def compile_model(self, model):
        model.compile(optimizer=Adam(lr=self.config.learning_rate),
                      loss=HierarchicalPartialLossModel.CumulatedCrossEntropy,
                      metrics=[HierarchicalPartialLossModel.CumulatedAccuracy, 'accuracy'])
        return model

    def train_model(self, model, train_data, validation_data):
        experiment_directory = self.create_experiment_directory()

        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        model_checkpoint = ModelCheckpoint(filepath=f'{experiment_directory}/best_hierarchical_model.h5',
                                           monitor='val_loss',
                                           save_best_only=True)


        history = model.fit(
            train_data[0],  # train_images
            train_data[1],  # train_labels
            validation_data=validation_data,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint],
        )

        print(f"Final training accuracy: {history.history['accuracy'][-1]}")
        print(f"Final validation accuracy: {history.history['val_accuracy'][-1]}")
        print(f"Final training custom metric: {history.history['CumulatedAccuracy'][-1]}")
        print(f"Final validation custom metric: {history.history['val_CumulatedAccuracy'][-1]}")

        if early_stopping.stopped_epoch > 0:
            print(f"Early stopped at epoch {early_stopping.stopped_epoch} with validation loss {early_stopping.best}")

        self.log_model_info(model, history, experiment_directory)
        return history

    def create_experiment_directory(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_directory = f"experiments/hierarchical/hierarchical_{self.config.pretrained_model.name}_{timestamp}"
        os.makedirs(experiment_directory, exist_ok=True)
        return experiment_directory

    def log_model_info(self, model, history, experiment_directory):
        with open(f"{experiment_directory}/model_info.txt", "w") as f:
            f.write("Hierarchical Partial Loss Model Information\n")
            f.write("==========================================\n\n")

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
            f.write(f"Final training custom metric: {history.history['CumulatedAccuracy'][-1]:.4f}\n")
            f.write(f"Final validation custom metric: {history.history['val_CumulatedAccuracy'][-1]:.4f}\n")

        # Plot training accuracy
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f"{experiment_directory}/accuracy.png")
        plt.close()

        # Plot training loss
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f"{experiment_directory}/loss.png")
        plt.close()

        # Plot custom metric
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['CumulatedAccuracy'])
        plt.plot(history.history['val_CumulatedAccuracy'])
        plt.title('Cumulated Accuracy')
        plt.ylabel('Cumulated Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig(f"{experiment_directory}/cumulated_accuracy.png")
        plt.close()

    @staticmethod
    def create_hierarchical_labels(target):
        # Ensure input is a TensorFlow tensor
        if not isinstance(target, tf.Tensor):
            target = tf.convert_to_tensor(target)

        # Create a copy of the input tensor to avoid modifying the original
        hierarchical = tf.identity(target)

        # Replace all 1s with 2s
        hierarchical = tf.where(tf.equal(hierarchical, 1), tf.constant(2, dtype=hierarchical.dtype), hierarchical)

        # Find positions of all 2s
        positions = tf.where(tf.equal(hierarchical, 2))

        # Add 1s next to 2s
        for pos in positions:
            pos1_int64 = tf.cast(pos[1], tf.int64)  # Cast to int64
            if pos1_int64 > 0:
                hierarchical = tf.tensor_scatter_nd_update(
                    hierarchical,
                    [[pos[0], pos1_int64 - 1]],
                    [tf.maximum(hierarchical[pos[0], pos1_int64 - 1], 1.0)]
                )
            if pos1_int64 < tf.shape(hierarchical, out_type=tf.int64)[1] - 1:  # Ensure shape is also int64
                hierarchical = tf.tensor_scatter_nd_update(
                    hierarchical,
                    [[pos[0], pos1_int64 + 1]],
                    [tf.maximum(hierarchical[pos[0], pos1_int64 + 1], 1.0)]
                )

        return hierarchical
    @staticmethod
    def CumulatedCrossEntropy(target, output, axis=-1):
        target_morphed = HierarchicalPartialLossModel.create_hierarchical_labels(target)
        mask_wide = clip_by_value(target_morphed, 0, 1)
        mask_narrow = clip_by_value(target_morphed, 1, 2) - 1

        output = tf.cast(output, tf.float32)
        output_sum = math_ops.reduce_sum(output, axis, keepdims=True)
        output_sum_epsilon = output_sum + tf.keras.backend.epsilon()
        output_normalized = output / output_sum_epsilon

        epsilon_ = tf.keras.backend.epsilon()
        return -math_ops.log(clip_ops.clip_by_value(math_ops.reduce_sum(mask_wide * output_normalized, axis), epsilon_, 1. - epsilon_)) + math_ops.log(clip_ops.clip_by_value(math_ops.reduce_sum(mask_narrow * output_normalized, axis), epsilon_,1. - epsilon_))

    @staticmethod
    def CumulatedAccuracy(y_true, y_pred, axis=-1):
        y_true_argmax = tf.argmax(y_true, axis)
        y_pred_argmax = tf.argmax(y_pred, axis)

        y_true_float = tf.cast(y_true_argmax, tf.float32)
        y_pred_float = tf.cast(y_pred_argmax, tf.float32)

        diff = tf.abs(y_true_float - y_pred_float)
        correct_predictions = tf.less_equal(diff, 1.0)

        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
        return accuracy

