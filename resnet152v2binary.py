# importlar
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.applications import ResNet152V2,InceptionResNetV2,DenseNet201,VGG19,Xception,EfficientNetB7,ResNet152
from tensorflow.keras.optimizers import Adam,SGD,RMSprop,Adagrad
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt




physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')



from tensorflow.keras.applications import ResNet50

# parametreler
MODEL_NAME = "deneme_ResNet152(epoch=30,val_accuracy= 0.7676,accuracy=0.7477,optimizer=Adam).h5"
TRAINING_SIZE = 17000
TEST_SIZE = 2000
IMG_SIZE = 64
IMG_CHANNELS = 3 # RGB
OPTIMIZER = "adam"
TRAINING_BATCH = 32
TEST_BATCH = 64
STEPS_PER_EPOCH = TRAINING_SIZE//TRAINING_BATCH
VALIDATION_STEPS = TEST_SIZE//TEST_BATCH
NUM_EPOCHS = 60
NUMBER_OF_CLASSES = 1


def build_model(backbone, lr=1e-4):
    model = Sequential()
    for layer in model.layers[:-15]:
      layer.trainable=False
    model.add(backbone)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1000, activation="relu",kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))  
    model.add(Dense(NUMBER_OF_CLASSES, activation="sigmoid",kernel_initializer='he_uniform'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=lr),
        metrics=['accuracy']
    )
    return model

resnet = ResNet152(
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
    )

test_datagen = ImageDataGenerator(
    rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset4/training_set',
                                           target_size=(IMG_SIZE, IMG_SIZE),
                                           batch_size=TRAINING_BATCH,                                           
                                           class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset4/test_set',
                                          target_size=(IMG_SIZE, IMG_SIZE),
                                          batch_size=TEST_BATCH,                                          
                                          class_mode='binary')

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


model.save(MODEL_NAME)


































