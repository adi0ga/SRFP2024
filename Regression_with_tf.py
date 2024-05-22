import tensorflow as tf
class densemod(tf.Module):
    def __init__(self,outf,name=None):
        super().__init__(name=name)
        self.outf=outf
        self.is_true=False
    def __call__(self,x):
        if not self.is_true:
            print(x.shape)
            self.inf=x.shape[-1]
            self.is_true=True
            self.w=tf.Variable(tf.random.normal([self.inf,self.outf]))
            self.b=tf.Variable(tf.random.normal([self.outf]))
        print(self.inf)
        print(self.outf)
        
        return tf.matmul(x,self.w) +self.b
class seqmod(tf.Module):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.dense1=densemod(outf=5)
        self.dense2=densemod(outf=3)
        self.dense3=densemod(outf=201)
    def __call__(self,x):
        x=self.dense1(x)
        x=self.dense2(x)
        x=self.dense3(x)
        return x
#initialize data
num_exp=201
vec=tf.linspace(-2.0,5.0,num_exp)
vec1=tf.linspace(-2.0,5.0,num_exp)
vec2=tf.linspace(-2.0,5.0,num_exp)
vec3=tf.linspace(-2.0,5.0,num_exp)
vec4=tf.linspace(-2.0,5.0,num_exp)
x=tf.Variable([vec,vec1,vec2,vec3,vec4])
W_ini=12.0
b_ini=3.0
def f(x):
    return W_ini*x+b_ini
noise=tf.random.normal(x.shape)
def loss(predicted,actual):
    return tf.sqrt(tf.reduce_sum(tf.square(actual-predicted)))
y=f(x)+noise
def train(model,x,y,learn):
    with tf.GradientTape() as tape:
        currentloss=loss(model(x),y)
    dw,db=tape.gradient(currentloss, [model.w,model.b])
    model.w.assign_sub(dw)
    model.b.assign_sub(db)
modelg=densemod(201)
def report(model,loss):
    return f"W={model.w.numpy():1.2f},b={model.b.numpy():1.3f},loss={loss:3.3f}"
def training_loop(epochs):
    for epoch in epochs:
        train(modelg,x,y,0.1)
        ep_loss=loss(modelg(x), y)
        print("Epoch",epoch)
        print(report(modelg,ep_loss))
epochs=range(10)
training_loop(epochs)