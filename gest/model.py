import argparse

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras_preprocessing.image import ImageDataGenerator

img_width, img_height = 150, 150
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def build():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model


def train(model, data_path, epochs):
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    train_generator = train_datagen.flow_from_directory(
        data_path,
        target_size=(img_height, img_width),
        class_mode='binary',
    )
    model.fit(train_generator, epochs=epochs)
    return model


def load(path):
    model = build()
    model.load_weights(path)
    return model


def score(model, frame):
    return model.predict(
        cv2.resize(frame, (img_width, img_height)).reshape((1, *input_shape)) / 255.
    )[0][0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", help="Data path")
    parser.add_argument("model_path", help="Model path")
    parser.add_argument("--load", help="Model path to load from")
    parser.add_argument("--epochs", help="Camera index", type=int, default=1)
    args = parser.parse_args()

    model = build() if args.load is None else load(args.load)
    model = train(model, args.data_path, args.epochs)
    model.save_weights(args.model_path)
