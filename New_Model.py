#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from keras.datasets import mnist
import numpy as np
dataset=mnist.load_data('mymnist.db')#Download MNIST Dataset
i=0
train,test=dataset#splitting of data into train and test data
x_train,y_train=train#Divide complete trainng set into training se for x and training dataset for y
x_test,y_test=test
x_train=x_train.reshape((x_train.shape[0],28,28,1))#Reshaping images into 28*28*1 bcz MNISTuse that size by default
x_test=x_test.reshape((x_test.shape[0],28,28,1))
x_train2=(x_train/255)-0.5#Normalisation/Scaling
x_test2=(x_test/255)-0.5
NUM_CLASSES=10 #Output among 10 categories can be possisble
y_train2=keras.utils.to_categorical(y_train,10)#10 possible Outputs for training
y_test2=keras.utils.to_categorical(y_test,10)#10 possible Outputs for testing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam,RMSprop,Adadelta
def make_model():
    model=Sequential()
    model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(28,28,1)))
    model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=None,padding='valid',data_format=None))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(64+i,activation='relu'))
    model.add(Dense(NUM_CLASSES))
    model.add(Activation('softmax'))
    return model
model=make_model()
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("mnist.h5",monitor="val_loss",mode="min",save_best_only = True,verbose=1)
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3,verbose = 1,restore_best_weights = True)
# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]
BATCH_SIZE=32
EPOCHS=1
if __name__=='__main__':
    model.fit(x_train2,y_train2, batch_size=BATCH_SIZE,epochs = EPOCHS,validation_data = (x_test2,y_test2),callbacks=callbacks,shuffle=True,verbose=1)
scores=model.evaluate(x_test2,y_test2,verbose=1)
#print('Test Loss:',scores[0])
print('TEST ACCURACY:',scores[1])


# In[ ]:




