import deepxde as xd
import numpy as np
import math
from matplotlib import pyplot as plt
w_n=4
zeta=0.1
#w_n=float(input("W_n:"))
#zeta=float(input("Zeta:"))
def ode(x,y):
    y_der=xd.grad.jacobian(y,x)
    y_dder=xd.grad.hessian(y,x)
    return y_dder + y_der*2*zeta*w_n+y*w_n**2
x0=0
v0=50
w_d=np.sqrt(1-zeta**2)*w_n
C=math.sqrt(x0**2+((zeta*w_n*x0+v0)/w_d)**2)
if w_d*x0==0:
    phi=math.pi/2
else:
    phi=math.atan(((zeta*w_n*x0)+v0)/w_d/x0)
def soln(t):
    exponent=np.exp(-zeta*w_n*t)
    return C*exponent*(np.cos((w_d*t)-phi))
geom=xd.geometry.TimeDomain(0,20)
def boundary(x,on_initial):
    return xd.utils.isclose(x[0],0)and on_initial
ic1=xd.icbc.IC(geom,lambda x:x0,boundary)
def error(inputs,outputs,X):
    return xd.grad.jacobian(outputs,inputs,i=0)-v0
ic2=xd.icbc.OperatorBC(geom,error, boundary)
data=xd.data.TimePDE(geom,ode,[ic1,ic2],3000,2,solution=soln,num_test=5000)
layer=[1]+5*[50]+[1]
activation="tanh"
initializer="Glorot normal"
net=xd.nn.FNN(layer,activation,initializer)
model=xd.Model(data,net)
model.compile("adam",lr=0.001,metrics=["l2 relative error"])
losshistory,train_state=model.train(iterations=50000)
xd.saveplot(losshistory, train_state)
T=np.linspace(0,20,1001).reshape(-1,1)
y_pred=model.predict(T)
plt.plot(T,y_pred,"--",label="Model")
plt.plot(T,soln(T),"-",label="True")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Displaacement")
plt.title("Single Degree of Freedom system")
plt.show()