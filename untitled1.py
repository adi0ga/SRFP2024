import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
x=tf.linspace(0.0,2*np.pi,101)
y=np.sin(x)
NOISE=tf.random.normal([101],stddev=0.2)
y1=y+NOISE
plt.plot(x,y1,".",label="data")
model=keras.Sequential([
    keras.layers.Input((1,)),
  #  keras.layers.Flatten(input_shape=(1,)),
    keras.layers.Dense(1,activation="gelu",name="layer_0"),
    keras.layers.Dense(50,activation="gelu",name="layer_1"),
    keras.layers.Dense(50,activation="gelu",name="layer_2"),
    keras.layers.Dense(50,activation="gelu",name="layer_3"),
    keras.layers.Dense(1,activation="gelu",name="layer_4")
    ])
loss_fn=keras.losses.MeanSquaredLogarithmicError()
model.compile(optimizer="adam",loss=loss_fn,metrics=["accuracy"])
model.fit(x,y1,epochs=1000)
y_new=model.predict(x)
plt.plot(x,y,"-",label="True Function")
plt.plot(x,y_new,"-",label="Model")