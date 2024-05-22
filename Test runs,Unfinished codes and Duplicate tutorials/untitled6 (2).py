# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:41:21 2024

@author: Adig1
"""
import tensorflow as tf
import keras
from keras import layers

# Define Sequential model with 3 layers
model=keras.Sequential(
    [keras.Input(shape=(2,)),
     layers.Dense(4, activation="relu"),
     layers.Dense(100,activation="relu"),
     layers.Dense(3,activation="relu")
     ])
x=tf.ones((3,2))
y=model(x)
print(y,"\n",model.weights[0].numpy(),"#########################",model.weights[2].numpy(),"###############",model.weights[5].numpy())


