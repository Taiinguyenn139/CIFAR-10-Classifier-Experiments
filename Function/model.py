from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dropout, Dense, Flatten, BatchNormalization, \
    Concatenate, Add, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, \
    LearningRateScheduler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time


class VGG:
    def __init__(self, num_classes, batch_size, epochs):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = None
        self.optimizer = Adam(learning_rate=0.001)
        self.NAME = 'VGG-Data-Augmentation-Drop-Out-BatchNorm-Batch-Size-1-{}'.format(int(time.time()))
        self.callbacks = [TensorBoard(log_dir='logs/{}'.format(self.NAME), histogram_freq=1)]

    def build(self):
        input_layer = Input(shape=(32, 32, 3))
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(
            input_layer)
        x = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.2)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(10, activation='softmax')(x)
        self.model = Model(inputs=input_layer, outputs=output)
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

    def fit(self, x_train, y_train, x_valid, y_valid):
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True,
                                     validation_split=0.2)
        datagen.fit(x_train)
        self.model.fit(datagen.flow(x_train, y_train, batch_size=32, subset='training'), batch_size=self.batch_size,
                       epochs=self.epochs,
                       validation_data=(x_valid, y_valid),
                       callbacks=self.callbacks)


