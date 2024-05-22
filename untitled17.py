import deepxde as xd
import numpy as np
k=float(input("Spring constant"))
c=float(input("Damping consatnt"))
m=float(input("mass"))
def ode(x,y):
    y_der=xd.grad.jacobian(y,x)
    y_dder=xd.grad.hessian(y,x)
    return (m*y_dder) + (y_der*c)+(y*k)
temp=(4*m*k)**(1/2)
w1=((4*m*k-c**2)**(1/2))/(2*m)
y0=float(input("y(0)"))
ydash0=float(input("y'(0)"))
def soln(t):
    if c<temp:
        c1=y0
        c2=(ydash0+((c1*c)/2*m))/w1
        return np.exp(-c*t/(2*m))*((c1*np.cos(w1*t))+(c2*np.sin(w1*t)))
    elif c==temp:
        c1=y0
        c2=ydash0+c*c1
        return np.exp(-c*t/(2*m))*(c1+(c2*t))
    elif c>temp:
        r1=(-c-(np.sqrt(c**2-4*m*k)))/(2*m)
        r2=(-c+np.sqrt(c**2-4*m*k))/(2*m)
        c1=(ydash0-r2*y0)/(r1-r2)
        c2=(ydash0-r1*y0)/(r2-r1)
        return (c1*np.exp(r1*t))+(c2*np.exp(r2*t))
geom=xd.geometry.TimeDomain(0,25)
def boundary(x,on_initial):
    return xd.utils.isclose(x[0], 0.0) and on_initial
ic1=xd.icbc.IC(geom,lambda x:y0,lambda _,on_initial: on_initial)
def error(inputs,outputs,X):
    return xd.grad.jacobian(outputs,inputs,i=0)-ydash0
ic2=xd.icbc.OperatorBC(geom,error, boundary)
data=xd.data.TimePDE(geom,ode,[ic1,ic2],600,1,solution=soln,num_test=100,train_distribution="LHS")
layer=[1]+[50]+[20]+[50]+[20]+[50]+[1]
activation="sigmoid"
initializer="Glorot normal"
net=xd.nn.FNN(layer,activation,initializer)
model=xd.Model(data,net)
model.compile("adam",lr=0.0089,metrics=["nanl2 relative error"])#,loss_weights=[0.01,1,1])
losshistory,train_state=model.train(iterations=10000)
xd.saveplot(losshistory, train_state)