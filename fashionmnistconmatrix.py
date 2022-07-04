# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:39:15 2022

@author: User
"""

from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import load_model
import os
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib import cm


import random

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()



print("x_train = ",x_train.shape)
print("x_test = ",x_test.shape)
print(x_train.shape[0],"eğitim örnekleri")
print(x_test.shape[0],"test örnekleri")



fashion_mnist_labels=np.array([
    'Tişört/Üst',
    'Pantolon',
    'Kazak',
    'Elbise',
    'Ceket',
    'Sandalet',
    'Gömlek',
    'Sneaker',
    'Çanta',
    'Bilekte Bot'
])


plt.imshow(x_train[10],cmap="gray_r")
plt.show()



from IPython.core.pylabtools import figsize
import numpy as np
fig,ax=plt.subplots(10,10,figsize=(10,10))
for i in range(10):
  for j in range(10):
    ax[i,j].imshow(x_train[np.random.randint(x_train.shape[0])],cmap="gray_r")
    ax[i,j].axis("off")
plt.show



x_train=x_train.reshape((60000,28*28))
x_train=x_train.astype("float32")/255

x_test=x_test.reshape((10000,28*28))
x_test=x_test.astype("float32")/255




from tensorflow.keras.utils import to_categorical

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)



from keras import models
from keras import layers
model=models.Sequential()
model.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
model.add(layers.Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',
              loss="categorical_crossentropy",
              metrics='accuracy')

history=model.fit(x_train,
                  y_train,
                  epochs=10,
                  batch_size=128,validation_split=0.2)


history_dict=history.history
print(history_dict)


#predic hesabı ve confusion matris
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
y_predict_fash = model.predict(x_test)
y_predict_fash=np.argmax(y_predict_fash, axis=1)
y_test_fash_eval=np.argmax(y_test, axis=1)

con_mat=confusion_matrix(y_test_fash_eval,y_predict_fash)
plt.style.use('seaborn-deep')
plt.figure(figsize=(10,10))
sns.heatmap(con_mat,annot=True,annot_kws={'size': 15},linewidths=0.5,fmt="d",cmap="gray")
plt.title('True or False predicted Fashion MNIST\n',fontweight='bold',fontsize=15)
plt.show()

#####################################################################################
#recall hesabı
import sklearn.metrics
recall = sklearn.metrics.recall_score(y_test_fash_eval,y_predict_fash, pos_label="positive",average='macro')
print(recall)
precision_score = sklearn.metrics.precision_score(y_test_fash_eval,y_predict_fash, pos_label="positive",average='macro')
print(precision_score)
f1_score = sklearn.metrics.f1_score(y_test_fash_eval,y_predict_fash, pos_label="positive",average='macro')
print(f1_score)

print('Classification report: \n ',sklearn.metrics.classification_report(y_test_fash_eval,y_predict_fash))

#####################################################################################
epochs=range(1,11)
loss=history_dict['loss']
accuracy=history_dict['accuracy']

plt.plot(epochs,loss)
plt.title("loss")
plt.xlabel("epochs")
plt.ylabel("eğitim kaybı")

plt.show()


plt.plot(epochs,accuracy)
plt.title("Accu")
plt.xlabel("Epochs")
plt.ylabel("eğitim başarısı")
plt.show()



test_loss,test_acc=model.evaluate(x_test,y_test)
print("test_loss = ",test_loss)
print("test_accuracy = ",test_acc)



print(history.history.keys())
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()






