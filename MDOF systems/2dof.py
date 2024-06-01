import deepxde as xd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from skopt.space import Integer,Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
"""
M=eval(input("enter mass matrix"))
M=np.array(M,dtype=np.float32)
K=eval(input("enter spring constant matrix"))
K=np.array(K,dtype=np.float32)
C=eval(input("enter damping matrix"))
C=np.array(C,dtype=np.float32)
F=eval(input("Enter driving force vector"))
F=np.array(F,dtype=np.float32)
y0=eval(input("Enter inital displacement vector"))
y0=np.array(y0,dtype=np.float32)
ydash0=eval(input("Enter inital velocity vector"))
ydash0=np.array(ydash0,dtype=np.float32)
"""
######
M = np.array([[1.0, 0.0], [0.0, 1.0]])
C = np.array([[0.1, 0.0], [0.0, 0.1]])
K = np.array([[2.0, -1.0], [-1.0, 2.0]])
F = np.array([1.0, 0.0])

#Domain
geom=xd.geometry.TimeDomain(0, 10)
#PINN
def multi_dof_system(t, u):
    u = u[:,0:2]
    u=tf.cast(u,tf.float64)
    v =tf.concat([xd.grad.jacobian(u, t,i=0),xd.grad.jacobian(u, t,i=1)],axis=1)
    v=tf.cast(v,tf.float64)
    a =tf.concat([xd.grad.jacobian(v, t,i=0),xd.grad.jacobian(v, t,i=1)],axis=1)
    a=tf.cast(a,tf.float64)
    return tf.cast(xd.utils.tf.matmul(a,M) + xd.utils.tf.matmul(v,C) + xd.utils.tf.matmul(u,K) - F,tf.float32)
def boundary(X,on_initial):
    return X[0]==0 and on_initial
def func(inputs,outputs,X):
    return tf.concat([xd.grad.jacobian(outputs, inputs,i=0),xd.grad.jacobian(outputs, inputs,i=1)],axis=1)
ic=xd.icbc.OperatorBC(geom, func, boundary)
"""
since a condition on the derivative cannot be set as a hard constarint.
I give it as an initial condition.Although the initial displacemnet is given as a hard constraint.

"""
data=xd.data.TimePDE(geom,multi_dof_system, [ic],3000,2,num_test=3000)
#defining Neural Netwrk
def build_model(config):
    iterations,num_layers,activation,n=config
    net=xd.nn.FNN([1]+num_layers*[50]+[2],activation,"Glorot normal")
    #transforming input to periodic data
    """
    A periodic input might help in this case as well as we expect a periodic output
    """
    def input_tranform(t):
        lst=[tf.sin(k*t)for k in range(1,n)]
        lst.insert(0,t)
        return tf.concat(lst,axis=1)
    #initial conditions
    """
    Initial displacement is set as a hard constraint
    """
    def hardconstraints(t,Y):
        y1=Y[:,0:1]
        y2=Y[:,1:2]
        return tf.concat([y1*tf.tanh(t),y2*tf.tanh(t)],axis=1)
    
    net.apply_feature_transform(input_tranform)
    net.apply_output_transform(hardconstraints)
    model=xd.Model(data,net)
    model.compile("adam",lr=0.001)
    return model
#model
dim_iterations=Integer(5000, 10000,name='iterations')
dim_num_layers=Integer(3, 15,name='num_layers')
dim_actiavtion=Categorical(["sin","tanh","sigmoid"],name="activation")
dim_n=Integer(2, 10,name="n")
ITERATION=0
dimensions=[dim_iterations,dim_num_layers,dim_actiavtion,dim_n]
@use_named_args(dimensions=dimensions)
def fitness(iterations,num_layers,activation,n):
    global ITERATION
    config=[iterations,num_layers,activation,n]
    print("Run No:",ITERATION+1,"\n",config)
    model=build_model(config)
    error=train_model(model, config)
    ITERATION+=1
    return error


"""Other features I have left the same as in the case of Lotka Volterra model"""
def train_model(model,config):
    iterations,num_layers,activation,n=config
    loss_history, train_state = model.train(iterations=iterations)
    model.compile("L-BFGS")
    loss_history, train_state = model.train()
    train_loss=np.array(loss_history.loss_train).sum(axis=1).ravel()
    test_loss=np.array(loss_history.loss_test).sum(axis=1).ravel()
    error=test_loss.min()
    return error
gp_minimize(fitness, dimensions,n_calls=15,x0=[5000,5,"sigmoid",4])