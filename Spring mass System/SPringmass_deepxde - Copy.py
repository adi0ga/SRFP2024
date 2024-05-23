import deepxde as xd
import numpy as np
import skopt
from skopt.plots import plot_convergence,plot_objective
from skopt.utils import use_named_args
from skopt.space import Integer,Real,Categorical
from skopt import gp_minimize
minimize="train loss"
k=float(input("Spring constant"))
c=float(input("Damping consatnt"))
m=float(input("mass"))
def ode(x,y):
    y_der=xd.grad.jacobian(y,x,i=0)
    y_dder=xd.grad.hessian(y,x,i=0)
    return m*y_dder + y_der*c+y*k
temp=(4*m*k)**(1/2)
w1=((4*m*k-c**2)**(1/2))/2/m
y0=float(input("y(0)"))
ydash0=float(input("y'(0)"))
c1=0
c2=0
def soln(t):
    if c<temp:
        global c1,c2
        c1=y0
        c2=(ydash0+((c1*c)/2/m))/w1
        return np.exp(-c*t/(2*m))*(c1*np.cos(w1*t)+c2*np.sin(w1*t))
    elif c==temp:
        c1=y0
        c2=ydash0+(c*c1/2/m)
        return np.exp(-c*t/(2*m))*(c1+c2*t)
    elif c>temp:
        r1=(-c-np.sqrt(c**2-4*m*k))/2/m
        r2=(-c+np.sqrt(c**2-4*m*k))/2/m
        c1=(ydash0-r2*y0)/(r1-r2)
        c2=(ydash0-r1*y0)/(r2-r1)
        return c1*np.exp(r1*t)+c2*np.exp(r2*t)
geom=xd.geometry.TimeDomain(0,10)
def boundary(x,on_initial):
    return xd.utils.isclose(x[0],0)and on_initial
ic1=xd.icbc.IC(geom,lambda x:c1,boundary)
def error(inputs,outputs,X):
    return xd.grad.jacobian(outputs,inputs,i=0)-ydash0
ic2=xd.icbc.OperatorBC(geom,error, boundary)


data=xd.data.TimePDE(geom,ode,[ic1,ic2],50,2,solution=soln,num_test=200)

def build_model(config):
    num_layers,size_layers,activation,lr,iterations=config
    layer=[1]+3*[50]+[1]
    initializer="Glorot normal"
    net=xd.nn.FNN(layer,activation,initializer)
    model=xd.Model(data,net)
    model.compile("adam",lr=0.001,metrics=["l2 relative error"])
    return model
def train_model(model,config):
    iterations=config[4]
    losshistory,train_state=model.train(iterations=iterations)
    test=np.array(losshistory.loss_test).sum(axis=1).ravel()
    train=np.array(losshistory.loss_train).sum(axis=1).ravel()
    metric=np.array(losshistory.metrics_test).sum(axis=1).ravel()
    if minimize=="test loss":
        error=test.min()
    elif minimize=="train loss":
        error=train.min()
    elif minimize=="test metric":
        error=metric.min()
    return error
dim_act=Categorical(["tanh","sigmoid"],name="activation")
dim_num_layers=Integer(3, 15,name="num_layers")
dim_size_layers=Integer(50, 150,name="size_layers")
dim_lr=Real(0.001, 0.01,name="lr")
dim_iterations=Integer(5000, 15000,name="iterations")
dimensions=[dim_num_layers,dim_size_layers,dim_act,dim_lr,dim_iterations]
@use_named_args(dimensions=dimensions)
def fitness(num_layers,size_layers,activation,lr,iterations):
    global ITERATION
    config=[num_layers,size_layers,activation,lr,iterations]
    print("Run No:",ITERATION+1)
    print(config)
    model=build_model(config)
    loss=train_model(model, config)
    if np.isnan(loss):
        loss=10**9       
    ITERATION+=1
    return loss
ITERATION=0
model_optim=gp_minimize(fitness, dimensions,n_calls=15,x0=[5,64,"sigmoid",0.005,10000])
print(model_optim.x)
plot_convergence(model_optim)
plot_objective(model_optim)
