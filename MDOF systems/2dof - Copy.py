import deepxde as xd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
#hyper parameters
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
net=xd.nn.FNN([1]+6*[50]+[2],"sin","Glorot normal")
#transforming input to periodic data
"""
A periodic input might help in this case as well as we expect a periodic output
"""
def input_tranform(t):
    return tf.concat([t,tf.sin(t),tf.sin(2*t),tf.sin(3*t),tf.sin(4*t),tf.sin(5*t),tf.sin(6*t)],axis=1)
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
#model
"""Other features I have left the same as in the case of Lotka Volterra model"""
model=xd.Model(data,net)
model.compile("adam",lr=0.001)
loss_history, train_state = model.train(iterations=5000)
model.compile("L-BFGS")
loss_history, train_state = model.train()
xd.saveplot(loss_history, train_state, issave=True, isplot=True)