import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten,Conv2D,MaxPooling2D
#import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist #to load datasets
(x_train ,y_train),(x_test ,y_test) = mnist.load_data() #x=images y=2

"""in order to train it we need to normalise it""" #28x28
x_train = tf.keras.utils.normalize(x_train ,axis=1)#B=0  W=255
x_test = tf.keras.utils.normalize(x_test ,axis=1)

#RESHAPING IMAGE FOR CONVOLUTIONAL OPERATION
print("\n\n RESHAPING FOR CNN\n\n")
IMG_SIZE = 28
x_trainr = np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1) #increasing the dimension for kernal=filter operation
x_testr = np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print("training sample dimension",x_trainr.shape)
print("testing sample dimension",x_testr.shape)

#CREATING DEEP LEARNING NETWORK
model = Sequential()

#first convolutional layer 28-3+1= 26x26
model.add(Conv2D(64,(3,3) ,input_shape = x_trainr.shape[1:])) #[1:] because we have (60000,28,28,1) but we need single image i,e (28,28,1)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2))) #gets only 2x2 matrix and rest will be removed

#second convolutional layer 26-3+1= 24x24
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

#third convolutional layer 24x24
model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

#fully connected layer #1
model.add(Flatten()) #2D to 1D
model.add(Dense(64)) #64 neurons [0,202,1221,255,,,,,,]
model.add(Activation('relu'))

#fully connected layer #2
model.add(Dense(32)) #32 neurons  we are reducing the no of neurons coz we are reaching 10
model.add(Activation('relu'))

#last fully connected layer
model.add(Dense(10)) #10 neurons represents 10 numbers from 0-9
model.add(Activation('softmax'))

model.summary()

print("Total training samples:",len(x_trainr))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train ,y_train ,epochs = 5 ,validation_split = 0.3)# for training

model.save('HR#3_neural.model') #Saving the model