# ====================================================================================
#  Author: Kunal SK Sukhija
# ====================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import missingno as msno
from sklearn import metrics
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
#%%
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.utils import to_categorical
#%%
(x_train,y_train),(x_test,y_true)=mnist.load_data()
#%%
x_train=x_train.reshape(-1,28,28,1)/255.0
x_test=x_test.reshape(-1,28,28,1)/255.0
y_train=to_categorical(y_train,num_classes=10)
y_true=to_categorical(y_true,num_classes=10)
#%%
model=Sequential()
#%%
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))
#%%
model.add(Flatten())
#%%
model.add(Dense(64,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
#%%
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
#%%
model.fit(x_train,y_train,batch_size=64,epochs=10,validation_split=0.1)
#%%
loss,accuracy=model.evaluate(x_test,y_true)
#%%
model.save('Digit_model.h5')