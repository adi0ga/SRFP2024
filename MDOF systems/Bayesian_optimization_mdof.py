import skopt
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real,Integer,Categorical
import deepxde as xd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skopt.plots import plot_convergence,plot_objective
# Define mass, damping, and stiffness matrices
M = np.array([[1.0, 0.0], [0.0, 1.0]])
C = np.array([[0.1, 0.0], [0.0, 0.1]])
K = np.array([[2.0, -1.0], [-1.0, 2.0]])
F = np.array([1.0, 0.0])

# Number of degrees of freedom
n = 2

def multi_dof_system(t, u):
    u = u[:,0:n]
    u=tf.cast(u,tf.float64)
    v =tf.concat([xd.grad.jacobian(u, t,i=0),xd.grad.jacobian(u, t,i=1)],axis=1)
    v=tf.cast(v,tf.float64)
    a =tf.concat([xd.grad.jacobian(v, t,i=0),xd.grad.jacobian(v, t,i=1)],axis=1)
    a=tf.cast(a,tf.float64)
    return tf.cast(xd.utils.tf.matmul(a,M) + xd.utils.tf.matmul(v,C) + xd.utils.tf.matmul(u,K) - F,tf.float32)
# Define initial conditions
def initial_conditions(X):
    u0 = np.zeros((n,))
    return u0
geom = xd.geometry.TimeDomain(0, 10)
def boundary(x,on_initial):
    return x[0]==0 and on_initial
def func(inputs,outputs,X):
    return tf.concat([xd.grad.jacobian(outputs,inputs,i=0),xd.grad.jacobian(outputs,inputs,i=1)],axis=1)
# Initial and boundary conditions
ic1 = xd.icbc.IC(geom, lambda X: initial_conditions(X),boundary ,component=0)
ic2 = xd.icbc.IC(geom, lambda X: initial_conditions(X), boundary,component=1)
ic3 = xd.icbc.OperatorBC(geom, func, boundary)
def model_build(config_a):
    num_layers,size_layers,activation,lr,optimizer=config_a
    data = xd.data.TimePDE(
        geom,
        multi_dof_system,
        [ic1,ic2 ,ic3],
        num_domain=100,num_test=500,
        num_boundary=2,
    #    num_initial=10
    )

    # Define neural network
    print(num_layers)
    net = xd.nn.FNN([1] + [int(size_layers)] * int(num_layers) + [n], activation, "Glorot normal")
    model = xd.Model(data, net)
    
    # Compile and train the model
    model.compile(optimizer, lr=lr,metrics=["accuracy"],loss_weights=[0.01,1,1,1])
    return model
def model_train(model):
    losshistory, train_state = model.train(iterations=10000)
    train=np.array(losshistory.loss_train).sum(axis=1).ravel()
    test=np.array(losshistory.loss_test).sum(axis=1).ravel()
    metric=np.array(losshistory.metrics_test).sum(axis=1).ravel()
    error=test.min()
    print(test)
    print(metric)
    print(train)
    return error
dim_num_layers=Integer(low=3, high=15,name="num_layers")
dim_size_layers=Integer(low=50,high=150,name="size_layers")
dim_activation=Categorical(["tanh","sin","sigmoid"],name="activation")
dim_lr=Real(0.001,0.005,name="lr")
dim_optimizer=Categorical(["adam","adadelta","sgd"],name="optimizer")
num_iter=12
dimensions=[dim_num_layers,dim_size_layers,dim_activation,dim_lr,dim_optimizer]
default_0=[3,50,"sigmoid",0.001,"adam"]
@use_named_args(dimensions=dimensions)
def fitness(num_layers,size_layers,activation,lr,optimizer):
    global ITERATION
    config_a=[int(num_layers),int(size_layers),activation,lr,optimizer]
    print("Run No:",ITERATION+1)
    print(config_a)
    model=model_build(config_a)
    error=model_train(model)
    if np.isnan(error):
        error=10**7
    ITERATION+=1
    return error
ITERATION=0
search_result=gp_minimize(func=fitness, dimensions=dimensions,acq_func="EI",n_calls=num_iter,x0=default_0,)
print(search_result.x)
plot_convergence(search_result)
plot_objective(search_result, show_points=True, size=3.8)
