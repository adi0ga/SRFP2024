import tensorflow as tf
class densemod(tf.Module):
    def __init__(self,outf,name=None):
        super().__init__(name=name)
        self.outf=outf
        self.is_true=False
    def __call__(self,x):
        if not self.is_true:
            self.inf=x.shape[-1]
            self.is_true=True
            self.w=tf.random.normal([self.inf,self.outf])
            self.b=tf.random.normal([self.outf])
        return tf.matmul(self.w,x)+self.b
class seqmod(tf.Module):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.dense1=densemod(outf=5)
        self.dense2=densemod(outf=3)
        self.dense3=densemod(outf=2)
    def __call__(self,x):
        x=self.dense1(x)
        x=self.dense2(x)
        x=self.dense3(x)
        return x
#initialize data
num_exp=2001
vec=tf.linspace(-2.0,5.0,num_exp)
W_ini=12.0
b_ini=3.0
def f(x):
    return W_ini*x+b_ini
noise=tf.random.normal(vec.shape)
def loss(predicted,actual):
    return tf.sqrt(tf.reduce_sum(tf.square(actual-predicted)))
y=f(vec)+noise
def train(model,x,y,learn):
    with tf.GradientTape() as tape:
        currentloss=loss(model(x),y)
    dw,db=tape.gradient(currentloss, [model.w,model.b])
    model.w.assign_sub(dw)
    model.b.assign_sub(db)
modelg=seqmod()
def report(model,loss):
    return f"W={model.w.numpy():1.3f},b={model.b.numpy():1.3f},loss={loss:1.3f}"
def training_loop(epochs):
    for epoch in epochs:
        train(modelg,vec,y,0.1)
        ep_loss=loss(modelg(vec), y)
        print("Epoch",epoch)
        print(report(modelg,ep_loss))
epochs=range(10)
training_loop(epochs)