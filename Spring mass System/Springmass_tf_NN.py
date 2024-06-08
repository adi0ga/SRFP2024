import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import math
x=tf.linspace(0.0,20,3001)
#Physical Cnstants
w_n=4
zeta=0.1
#Initial Conditions
x0=0
v0=50
#Analytical Solution
w_d=np.sqrt(1-zeta**2)*w_n
C=math.sqrt(x0**2+((zeta*w_n*x0+v0)/w_d)**2)
if w_d*x0==0:
    phi=math.pi/2
else:
    phi=math.atan(((zeta*w_n*x0)+v0)/w_d/x0)
def soln(t):
    exponent=np.exp(-zeta*w_n*t)
    return C*exponent*(np.cos((w_d*t)-phi))

y=soln(x)
#NOISE=tf.random.normal([101],stddev=0.2)
y1=y#+NOISE
plt.plot(x,y1,".",label="data")
class NNModel(tf.keras.Model):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.layer0=tf.keras.layers.Flatten()
        self.layer1=tf.keras.layers.Dense(1,input_shape=(5001,),activation="tanh")
        self.layer2=tf.keras.layers.Dense(50,activation="tanh")
        self.layer3=tf.keras.layers.Dense(50,activation="tanh")
        self.layer4=tf.keras.layers.Dense(50,activation="tanh")
        self.layer5=tf.keras.layers.Dense(1,activation="tanh")
        
    def __call__(self,x):
        x=self.layer0(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return self.layer5(x)
model=NNModel()
loss_fn=keras.losses.MeanSquaredError()
optimizer=keras.optimizers.Adam(learning_rate=0.001)
train_loss=tf.keras.metrics.Mean(name="train_loss")
train_accuracy=tf.keras.metrics.Accuracy(name="train_accuracy")
predictions={}
losses={}
def train(x,y):
    with tf.GradientTape() as tape:
        prediction=model(x)
        predictions[epoch]=prediction
        loss=loss_fn(y,prediction)
    
    gradient=tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient,model.trainable_variables))
    losses[epoch]=train_loss(loss).numpy()

    if epoch%6000==0:
        print("TRAIN loss:",train_loss(loss))
        print("TRAIN accuracy:",train_accuracy(y,prediction))
        plt.plot(x,prediction,"-",label=f"epoch_{epoch}")
        if epoch%18000==0:
            plt.legend()
            plt.show()
epochs=60000
for epoch in range(epochs):
    global brk_val
    brk_val=epoch
    if epoch%5000==0:
        print("epoch$",epoch,"of",epochs)
    train(x,y1)
    if float(losses[epoch])<=0.02:
        brk_val=epoch
        break


plt.plot(x,y1,".",label="data")
plt.plot(x,y,"-",label="TRUE")
plt.plot(x,predictions[brk_val],"-",label="Model")
plt.legend()
plt.show()