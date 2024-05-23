import tensorflow as tf
import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

TRUE_W=4.0
true_q=3.0
True_b=2.0
num_ex=201
x=tf.linspace(-2.0,2.0,num_ex)
vec=tf.linspace(-2.0,2.0,int(num_ex/2))
x=tf.cast(x, tf.float32)
vec=tf.cast(vec,tf.float32)
def f(x):
    return (x**2)*TRUE_W+x*true_q+True_b

noise=tf.random.normal(shape=[num_ex],stddev=5.0)

y=f(x)+noise
plt.plot(x,y,".")
plt.show()

class Mymodel(tf.Module):
    def __init__(self,**kwargs):
        self.w=tf.Variable(-7.0)
        self.b=tf.Variable(2.0)
        self.q=tf.Variable(10.0)
    def __call__(self,x):
        return self.w*(x**2)+self.q*x+self.b
model=Mymodel()

print(model.variables)

def loss(target_y,predicted_y):
    return tf.reduce_mean(tf.square(target_y-predicted_y))

plt.plot(x,y,".",label="Data")
plt.plot(x,f(x), label="Ground Truth")
plt.plot(x,model(x),label="Predictions")
plt.legend()
plt.show()
print("current loss:%1.6f"%loss(y,model(x)).numpy())

def train(model,x,y,learning_rate):
    with tf.GradientTape() as tape:
        current_loss=loss(y,model(x))
    dw,db,dq=tape.gradient(current_loss,[model.w,model.b,model.q])
    model.w.assign_sub(learning_rate*dw)
    model.b.assign_sub(learning_rate*db)
    model.q.assign_sub(learning_rate*dq)
model=Mymodel()
weights=[]
biases=[]
quants=[]
epochs=range(200)
def report(model,loss):
    return f"W={model.w.numpy():1.2f},b={model.b.numpy():1.2f},loss={loss:2.5f}"
def training_loop(model,x,y):
    for epoch in epochs:
        train(model,x,y,learning_rate=0.01)
        quants.append(model.q.numpy())
        weights.append(model.w.numpy())
        biases.append(model.b.numpy())
        current_loss=loss(y,model(x))
        print(f"Epoch{epoch:2d}:")
        print("       ",report(model,current_loss))
current_loss=loss(y,model(x))
print("*********")
print("   ",report(model,current_loss)) 
training_loop(model, x, y)       




plt.plot(epochs, weights, label='Weights', color=colors[0])
plt.plot(epochs, [TRUE_W] * len(epochs), '--',
         label = "True weight", color=colors[0])

plt.plot(epochs, biases, label='bias', color=colors[1])
plt.plot(epochs, [True_b] * len(epochs), "--",
         label="True bias", color=colors[1])

plt.plot(epochs, quants, label='quants', color=colors[2])
plt.plot(epochs, [true_q] * len(epochs), "--",
         label="True quants", color=colors[2])
plt.legend()
plt.show()


plt.plot(x, y, '.', label="Data")
plt.plot(x, f(x), label="Ground truth")
plt.plot(x, model(x), label="Predictions")
plt.legend()
plt.show()

print("Current loss: %1.6f" % loss(model(x), y).numpy())


