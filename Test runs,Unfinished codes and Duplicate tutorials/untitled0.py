import tensorflow as tf
from Input_Info import analytical_solution
import tensorflow as tf
import keras
from keras import layers
import numpy
# Define the problem parameters
m = 1.0  # Mass
k = 1  # Spring constant
x0 = 0.5  # Initial displacement
v0 = 1 # Initial velocity
c= 0.5  #damping coefficient
# Define the training configuration
t_start = 0.0
t_end = 10.0
num_points = 1000
epochs = 5000
learning_rate = 0.1
x=tf.linspace(0.0, 10.0, 1001)
x1=tf.Variable(x)
x2=numpy.transpose(x1.numpy())
y=analytical_solution(x2, m, c, k, x0, v0)
y1=tf.Variable(y)
"""class densemod(tf.Module):
    def __init__(self,input_size,output_size,name=None):
        super().__init__(name=name)
        W=tf.normal.random([input_size,output_size])
        b=tf.normal.random(output_size)
    def __call__(self,input_data):
        return tf.matmul(self.W,input_data)+self.b
"""
model=keras.Sequential([
    layers.Dense(50,activation="relu"),
    ])
x2.reshape((1,1001))
model.add(layers.Dense(50,activation="relu"))
model.add(layers.Dense(50,activation="relu"))
loss_fn=tf.keras.losses.SparseCategoricalCrossentropy
model.compile(optimizer="adam",loss=loss_fn,metrics=['accuracy'])
model.fit(x2, y, epochs=5000)
"""
def Train(data,hidden_size,n_hidden):
    layers={f"layer{n_hidden}":densemod(hidden_size,1)}
    for j in range(n_hidden):
        if j==0:
            layers[f'layer_{j}']=densemod(1,hidden_size)
        else:
            layers[f'layer_{j}']=densemod(hidden_size,hidden_size)
    train=[]
    loss=[]
    for i in range(n_hidden+1):
        if i==0: train.append(data["input"])
        train.append(tf.nn.relu(layers[f'layer_{i}'](train[i])))
    for k in range(n_hidden-1,-1,-1):
        if k==n_hidden:
            loss[n_hidden-1]=data["output"]-train[]
            
    return train[-1]


"""