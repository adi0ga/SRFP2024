import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
x=tf.linspace(0.0,2*np.pi,101)
y=np.sin(x)
NOISE=tf.random.normal([101],stddev=0.2)
y1=y+NOISE
plt.plot(x,y1,".",label="data")
class NNModel(tf.keras.Model):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.layer0=tf.keras.layers.Flatten()
        self.layer1=tf.keras.layers.Dense(1,input_shape=(101,),activation="tanh")
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

    if epoch%600==0:
        print("TRAIN loss:",train_loss(loss))
        print("TRAIN accuracy:",train_accuracy(y,prediction))
        plt.plot(x,prediction,"-",label=f"epoch_{epoch}")
epochs=6000
for epoch in range(epochs):
    if epoch%200==0:
        print("epoch$",epoch,"of",epochs)
    train(x,y1)
    if float(losses[epoch])<=0.12:
        brk_val=epoch
        break
plt.legend()
plt.show()

plt.plot(x,y1,".",label="data")
plt.plot(x,y,"-",label="TRUE")
plt.plot(x,predictions[brk_val],"-",label="Model")
plt.legend()
plt.show()