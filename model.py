# importlar
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import Xception,ResNet152,DenseNet201,InceptionResNetV2

physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')


# parametreler
MODEL_NAME = "deneme_DenseNet201(3hastalık,20epoch,).h5"
TRAINING_SIZE = 11400
TEST_SIZE = 1200
IMG_SIZE = 75
IMG_CHANNELS = 3 # RGB
OPTIMIZER = "adam"
TRAINING_BATCH = 32
TEST_BATCH = 64
STEPS_PER_EPOCH = TRAINING_SIZE//TRAINING_BATCH
VALIDATION_STEPS = TEST_SIZE//TEST_BATCH
NUM_EPOCHS = 20
NUMBER_OF_CLASSES = 3


def build_model(backbone, lr=1e-4):
    model = Sequential()
    for layer in model.layers[:-15]:
      layer.trainable=False
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(NUMBER_OF_CLASSES, activation="softmax"))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    return model


resnet = DenseNet201(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

model = build_model(resnet, lr=1e-4)
model.summary()



# Data Augmentation
train_datagen = ImageDataGenerator(
    #yeniden boyutlandırma
    rescale=1./255,
    #kesme,kayma
    shear_range=0.2,
    #yakinlastirma
    zoom_range=0.2,
    #yatay cevirme
    horizontal_flip=True,
    #dondurme
    )

test_datagen = ImageDataGenerator(
    rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset1/training_set',
                                           target_size=(IMG_SIZE, IMG_SIZE),
                                           batch_size=TRAINING_BATCH,                                           
                                           class_mode='categorical')

test_set = test_datagen.flow_from_directory('dataset1/test_set',
                                          target_size=(IMG_SIZE, IMG_SIZE),
                                          batch_size=TEST_BATCH,                                         
                                          class_mode='categorical')

histories=model.fit_generator(training_set,
                             steps_per_epoch=STEPS_PER_EPOCH,
                             epochs=NUM_EPOCHS,
                             validation_data=test_set,
                             validation_steps=VALIDATION_STEPS)

plt.plot(histories.history['accuracy'])
plt.plot(histories.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


plt.plot(histories.history['loss'])
plt.plot(histories.history['val_loss'])
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


#model.save(MODEL_NAME)


































