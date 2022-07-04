# importlar
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,MaxPooling2D,Dropout
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt

# parametreler
MODEL_NAME = "deneme1.h5"
TRAINING_SIZE = 11400
TEST_SIZE = 1200
IMG_SIZE = 64
IMG_CHANNELS = 3
OPTIMIZER = "adam"
TRAINING_BATCH = 32
TEST_BATCH = 64
STEPS_PER_EPOCH = TRAINING_SIZE//TRAINING_BATCH
VALIDATION_STEPS = TEST_SIZE//TEST_BATCH
NUM_EPOCHS = 10
NUMBER_OF_CLASSES = 3

def model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(NUMBER_OF_CLASSES, activation='softmax'))
    model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# model
classifier = model()
classifier.summary()

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
    rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset1/training_set',
                                           target_size=(IMG_SIZE, IMG_SIZE),
                                           batch_size=TRAINING_BATCH,
                                           class_mode='categorical',
                                           shuffle = True)

test_set = test_datagen.flow_from_directory('dataset1/test_set',
                                          target_size=(IMG_SIZE, IMG_SIZE),
                                          batch_size=TEST_BATCH,
                                          class_mode='categorical',
                                          shuffle = True)

classifier.fit_generator(training_set,
                             steps_per_epoch=STEPS_PER_EPOCH,
                             epochs=NUM_EPOCHS,
                             validation_data=test_set,
                             validation_steps=VALIDATION_STEPS)

# classifier.save(MODEL_NAME)






















