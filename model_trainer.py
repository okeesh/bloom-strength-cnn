from keras.applications import MobileNet
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout

from keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

from load_data import load_data
from model import ModelTrainer
from model_config import ModelConfig

config = ModelConfig(
    model_type='regression',
    learning_rate=0.001,
    dropout_rate=0.3,
    batch_size=16,
    epochs=50,
    pretrained_model=MobileNet(weights='imagenet', include_top=False),
    dense_units=512
)


model = ModelTrainer(config)

# Load train and validation data from load_data
train_images, train_labels, validation_images, validation_labels, class_weights = load_data(model_type='regression')
train_data = (train_images, train_labels)
validation_data = (validation_images, validation_labels)
model.train(train_data, validation_data)

