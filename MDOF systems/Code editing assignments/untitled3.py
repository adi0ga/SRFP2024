
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.linalg import eig,inv
# Number of degrees of freedom
n = 2

# Define mass, damping, and stiffness matrices
M = np.array([[1.0, 0.0], [0.0, 2.0]])
C = np.array([[1.6, -0.8], [-0.8, 0.8]])
K = np.array([[5.0, 0.0], [-4.0, 4.0]])
F = np.array([1.0, 0.0])
u0=np.zeros((n,))
v0=np.zeros((n,))
"""
def multi_dof_system(t, u):
    u = u[:,0:n]
    u=tf.cast(u,tf.float64)
    v =tf.concat([dde.grad.jacobian(u, t,i=0),dde.grad.jacobian(u, t,i=1)],axis=1)
    v=tf.cast(v,tf.float64)
    a =tf.concat([dde.grad.jacobian(v, t,i=0),dde.grad.jacobian(v, t,i=1)],axis=1)
    a=tf.cast(a,tf.float64)
    return tf.cast(dde.utils.tf.matmul(a,M) + dde.utils.tf.matmul(v,C) + dde.utils.tf.matmul(u,K) - F,tf.float32)
# Define initial conditions
def initial_conditions(X):
    u0 = np.zeros((n,))
    #v0 = np.zeros((n,))
   # return np.concatenate([u0, v0])
    return u0
geom = dde.geometry.TimeDomain(0, 50)
def boundary(x,on_initial):
    return dde.utils.isclose(x[0], 0) and on_initial
def func(inputs,outputs,X):
    return tf.concat([dde.grad.jacobian(outputs,inputs,i=0),dde.grad.jacobian(outputs,inputs,i=1)],axis=1)
# Initial and boundary conditions
ic1 = dde.icbc.IC(geom, lambda X: initial_conditions(X),boundary ,component=0)
ic2 = dde.icbc.IC(geom, lambda X: initial_conditions(X), boundary,component=1)
ic3 = dde.icbc.OperatorBC(geom, func, boundary)

data = dde.data.TimePDE(
    geom,
    multi_dof_system,
    [ic1,ic2 ,ic3],
    num_domain=3000,num_test=5000,
    num_boundary=2,
#    num_initial=10
)
def input_tranform(t):
    return tf.concat([t,tf.sin(t),tf.sin(2*t),tf.sin(3*t),tf.sin(4*t),tf.sin(5*t),tf.sin(6*t)],axis=1)

# Define neural network
net = dde.nn.FNN([1] + [50]*5 + [n], "sin", "Glorot normal")
net.apply_feature_transform(input_tranform)
""""""def hardconstraints(t,Y):
    r=Y[:,0:1]
    p=Y[:,1:2]
    return tf.concat([r*tf.tanh(t),p*tf.tanh(t)],axis=1)

net.apply_output_transform(hardconstraints)"""
"""
model = dde.Model(data, net)
# Compile and train the model
model.compile("adam", lr=0.001,metrics=[],loss_weights=[1,1,1,1])
losshistory, train_state = model.train(iterations=20000)
"""""""model.compile("L-BFGS")
losshistory, train_state = model.train()
"""""""
dde.saveplot(losshistory, train_state)
# Evaluate the model
X = np.linspace(0, 50, 1001).reshape(-1, 1)
y_pred = model.predict(X)

from scipy.linalg import inv
x0=0
x1=50
num=1000
t=np.linspace(x0,x1, num+1).reshape(-1,1)
v=np.zeros((n,num+1))
a=np.zeros((n,num+1))
u=np.zeros((n,num+1))
dt=(x1-x0)/num
for j in range(0,n):
    u[j][0]=u0[j]
    v[j][0]=v0[j]
    a[j][0]=(inv(M)@(F-v0@C-u0@K))[j]
    for i in range(1,num+1):
        v[j][i]=(a[j][i-1]*dt)+v[j][i-1]
        u[j][i]=((a[j][i-1]*(dt)**(2))/2)+(v[j][i-1] *dt)+u[j][i-1]
        a[j][i]=(inv(M)@(F-v[:,i]@C-u[:,i]@K))[j]

# Plot the results
#plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(X, y_pred[:, i], "--",label=f'DOF {i+1}')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement vs. Time for Multiple DOF System')
plt.grid(True)

plt.plot(t,u[0],"-",label="true_1",color="black")
plt.plot(t, u[1],"-",label="true_2")
plt.legend()
plt.show()
"""
##############################################
A0=np.zeros((n,n))
A1=np.identity(n)
A2=-inv(M)@K
A3=-inv(M)@C
A=np.concatenate((A0,A1),axis=1)
A2=np.concatenate((A2,A3),axis=1)
A=np.concatenate((A,A2),axis=0)
B=np.concatenate((A0,inv(M)),axis=0)
w,y,x=eig(A,left=True)
Result=list(np.zeros(2*n,))
for r in range(2*n):
    Result[r]=((np.transpose(y)[r,:]@B@F)*(x[:,r]))/-w[r]
Solution=sum(Result)