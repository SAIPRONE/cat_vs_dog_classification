# -*- coding: utf-8 -*-
"""
@author: Fadi Helal
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split

# Define constants
FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3
TRAIN_DATA_PATH = "C:/Users/saibr/Desktop/ML/supervised-learning/jamk-cnn/input/train/train/"
TEST_DATA_PATH = "C:/Users/saibr/Desktop/ML/supervised-learning/jamk-cnn/input/test1/test1/"

# Load the data
def load_data():
    filenames = os.listdir(TRAIN_DATA_PATH)
    categories = ['dog' if filename.split('.')[0] == 'dog' else 'cat' for filename in filenames]
    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })
    df['category'] = df['category'].replace({'cat': 0, 'dog': 1})
    return df

# Plot sample image
def plot_sample_image(df):
    sample = random.choice(df['filename'].values)
    image = load_img(TRAIN_DATA_PATH+sample)
    plt.imshow(image)
    plt.show()

# Create model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

# Reduce learning rate when accuracy is not improving
def create_learning_rate_reduction():
    return ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)

# Split data into train and validate datasets
def split_data(df):
    train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
    train_df = train_df.reset_index(drop=True)
    validate_df = validate_df.reset_index(drop=True)
    return train_df, validate_df

# Create data generators
def create_data_generators(train_df, validate_df):
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        TRAIN_DATA_PATH, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=15
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        TRAIN_DATA_PATH, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=15
    )

    return train_generator, validation_generator

# Train the model
def train_model(model, train_generator, validation_generator, learning_rate_reduction):
    epochs = 3 if FAST_RUN else 10
    history = model.fit_generator(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=total_validate//batch_size,
        steps_per_epoch=total_train//batch_size,
        callbacks=[learning_rate_reduction]
    )
    return history

# Plot the accuracy and loss graphs
def plot_graphs(history, epochs):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(history.history['loss'], color='b', label="Training loss")
    ax1.plot(history.history['val_loss'], color='r', label="validation loss")
    ax1.set_xticks(np.arange(1, epochs, 1))
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
    ax2.plot(history.history['val_accuracy'], color='r', label="Validation accuracy")
    ax2.set_xticks(np.arange(1, epochs, 1))

    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

# Start of the main script
if __name__ == "__main__":
    df = load_data()
    plot_sample_image(df)
    model = create_model()
    learning_rate_reduction = create_learning_rate_reduction()
    train_df, validate_df = split_data(df)
    train_generator, validation_generator = create_data_generators(train_df, validate_df)
    history = train_model(model, train_generator, validation_generator, learning_rate_reduction)
    plot_graphs(history, 3 if FAST_RUN else 10)
