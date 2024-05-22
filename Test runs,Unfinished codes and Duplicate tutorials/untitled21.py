import deepxde as xd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
#hyper parameters
R=20
U=200
#Domain
geom=xd.geometry.TimeDomain(0, 10)
#diff eqn
def func(t,r):
    x,y=r
    dx_t=1/U*R*(2.0*U*x-0.04*U*x*U*y)
    dy_t=1/U*R*(0.02*U*x*U*y-1.06*U*y)
    return dx_t,dy_t
def ode(t,Y):
    r=Y[:,0:1]
    p=Y[:,1:2]
    r_der=xd.grad.jacobian(r, t)
    p_der=xd.grad.jacobian(p,t)
    return tf.concat([r_der-(2*R*r)+(0.04*U*R*p*r),p_der+(1.06*R*p)-(0.02*U*R*p*r)],axis=1)
def initial(x,On_initial):
    return xd.utils.isclose(x[0],0) and On_initial
ic1=xd.icbc.IC(geom,lambda X:100/U,initial,component=0)
ic2=xd.icbc.IC(geom,lambda X:15/U,initial,component=1)

data=xd.data.TimePDE(geom,ode, [ic1,ic2],100,2,num_test=2000)
#defining model
net=xd.nn.FNN([1]+8*[50]+[2],"tanh","Glorot normal")
model=xd.Model(data,net)
model.compile("adam",0.001)
loss_history,train_state=model.train(iterations=10000)
xd.saveplot(loss_history, train_state)