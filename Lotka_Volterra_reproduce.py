import deepxde as xd
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
#hyper parameters
R=20
U=200
#Domain
geom=xd.geometry.TimeDomain(0, 1)
#diff eqn for data generation
def func(t,x):
    r,p=x
    dr_t=1/U*R*(2.0*U*r-0.04*U*r*U*p)
    dp_t=1/U*R*(0.02*U*r*U*p-1.06*U*p)
    return dr_t,dp_t
#PINN
def ode(t,Y):
    r=Y[:,0:1]
    p=Y[:,1:2]
    r_der=xd.grad.jacobian(r, t)
    p_der=xd.grad.jacobian(p,t)
    return tf.concat([r_der-(2*R*r)+(0.04*U*R*p*r),p_der+(1.06*R*p)-(0.02*U*R*p*r)],axis=1)
data=xd.data.TimePDE(geom,ode, [],3000,2,num_test=3000)
#defining Neural Netwrk
net=xd.nn.FNN([1]+8*[50]+[2],"tanh","Glorot normal")
#transforming input to periodic data
def input_tranform(t):
    return tf.concat([t,tf.sin(t),tf.sin(2*t),tf.sin(3*t),tf.sin(4*t),tf.sin(5*t),tf.sin(6*t)],axis=1)
#initial conditions
def hardconstraints(t,Y):
    r=Y[:,0:1]
    p=Y[:,1:2]
    return tf.concat([r*tf.tanh(t)+100/U,p*tf.tanh(t)+15/U],axis=1)
net.apply_feature_transform(input_tranform)
net.apply_output_transform(hardconstraints)
#model
model=xd.Model(data,net)
model.compile("adam",lr=0.001)
loss_history, train_state = model.train(iterations=5000)
model.compile("L-BFGS")
loss_history, train_state = model.train()
xd.saveplot(loss_history, train_state, issave=True, isplot=True)

#############################
#generating original data####
#############################
t=np.linspace(0,1,101)
def generate_data():
    t=np.linspace(0,1,101)
    solution=integrate.solve_ivp(func, (0,10),(100/U,15/U) , t_eval=t)
    r_true,p_true=solution.y
    r_true = r_true.reshape(101, 1)
    p_true = p_true.reshape(101, 1)

    return r_true, p_true
r_true, p_true = generate_data()
plt.plot(t, r_true, color="black", label="r_true")
plt.plot(t, p_true, color="blue", label="p_true")
t.reshape((101,1))
t=np.expand_dims(t, axis=-1)
prediction=model.predict(t)
r_predict=prediction[:,0:1]
p_predict=prediction[:,1:2]
plt.plot(t, r_predict, color="red", linestyle="dashed", label="r_pred")
plt.plot(t, p_predict, color="orange", linestyle="dashed", label="p_pred")
plt.legend()
plt.show()
