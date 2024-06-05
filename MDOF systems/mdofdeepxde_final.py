
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
m=5
k=2
q0=1
# Define mass, damping, and stiffness matrices
M = m*np.array([[1.0, 0.0, 0.0], [0.0, 1.0,0.0],[0.0, 0.0,2.0]])
C = np.array([[0.0, 0.0,0.0],[0.0, 0.0,0.0],[0.0, 0.0,0.0]])
K = k*np.array([[2.0, -1.0, 0.0], [-1.0, 3.0,-2.0],[0.0, -2.0,2.0]])
F = np.array([0.0, 0.0,0.0])
# Number of degrees of freedom
n = 3

def multi_dof_system(t, u):
    u = u[:,0:n]
    u=tf.cast(u,tf.float64)
    v =tf.concat([dde.grad.jacobian(u, t,i=k) for k in range(n)],axis=1)
    v=tf.cast(v,tf.float64)
    a =tf.concat([dde.grad.jacobian(v, t,i=k) for k in range(n)],axis=1)
    a=tf.cast(a,tf.float64)
    return tf.cast(dde.utils.tf.matmul(a,M) + dde.utils.tf.matmul(v,C) + dde.utils.tf.matmul(u,K) - F,tf.float32)
# Define initial conditions
def initial_conditions(X):
    u0 = np.zeros((n,))
    #v0 = np.zeros((n,))
   # return np.concatenate([u0, v0])
    return u0
geom = dde.geometry.TimeDomain(0, 10)
def boundary(x,on_initial):
    return x[0]==0 and on_initial
def func(inputs,outputs,X):
    return tf.concat([dde.grad.jacobian(outputs,inputs,i=0),dde.grad.jacobian(outputs,inputs,i=1)],axis=1)
# Initial and boundary conditions
ic1 = dde.icbc.IC(geom, lambda X: 1*q0,boundary ,component=0)
ic2 = dde.icbc.IC(geom, lambda X: 2*q0, boundary,component=1)
ic3 = dde.icbc.IC(geom, lambda X: 3*q0, boundary,component=2)
ic4 = dde.icbc.OperatorBC(geom, func, boundary)

data = dde.data.TimePDE(
    geom,
    multi_dof_system,
    [ic1,ic2 ,ic3,ic4],
    num_domain=5000,num_test=5000,
    num_boundary=2,
#    num_initial=10
)

# Define neural network
net = dde.nn.FNN([1] + [50] * 5 + [n], "tanh", "Glorot normal")
model = dde.Model(data, net)

# Compile and train the model
model.compile("adam", lr=0.005,metrics=["accuracy"],loss_weights=[0.01,1,1,1,1])
losshistory, train_state = model.train(iterations=25000)
dde.saveplot(losshistory, train_state)
# Evaluate the model
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(X, y_pred[:, i], label=f'DOF {i+1}')
def sol1(t):
    y1=(1.2812*np.cos(0.3731*(np.sqrt(k/m)*t)))-(0.4132*np.cos(1.3213*np.sqrt(k/m)*t))+(0.1320*np.cos(2.0285*np.sqrt(k/m)*t))
    return y1
def sol2(t):
    y1=(2.3841*np.cos(0.3731*np.sqrt(k/m)*t))-(0.1050*np.cos(1.3213*np.sqrt(k/m)*t))-(0.2791*np.cos(2.0285*np.sqrt(k/m)*t))
    return y1
def sol3(t):
    y1=(2.7696*np.cos(0.3731*np.sqrt(k/m)*t))+(0.1408*np.cos(1.3213*np.sqrt(k/m)*t))+(0.0896*np.cos(2.0285*np.sqrt(k/m)*t))
    return y1
t=np.linspace(0,10,1001)
plt.plot(t,sol1(t),"--",label="True_1")
plt.plot(t,sol2(t),"--",label="True_2")
plt.plot(t,sol3(t),"--",label="True_3")
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement vs. Time for Multiple DOF System')
plt.legend()
plt.grid(True)
plt.show()