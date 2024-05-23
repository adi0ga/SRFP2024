
import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

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
geom = dde.geometry.TimeDomain(0, 10)
def boundary(x,on_initial):
    return x[0]==0 and on_initial
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
    num_domain=100,num_test=500,
    num_boundary=2,
#    num_initial=10
)

# Define neural network
net = dde.nn.FNN([1] + [100] * 13 + [n], "sin", "Glorot normal")
model = dde.Model(data, net)

# Compile and train the model
model.compile("adam", lr=0.0019515584134263103,metrics=["accuracy"],loss_weights=[0.01,1,1,1])
losshistory, train_state = model.train(iterations=10000)
dde.saveplot(losshistory, train_state)
# Evaluate the model
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred = model.predict(X)

# Plot the results
plt.figure(figsize=(10, 6))
for i in range(n):
    plt.plot(X, y_pred[:, i], label=f'DOF {i+1}')
plt.xlabel('Time')
plt.ylabel('Displacement')
plt.title('Displacement vs. Time for Multiple DOF System')
plt.legend()
plt.grid(True)
plt.show()
