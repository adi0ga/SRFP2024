import numpy as np
import deepxde as xd
k=eval(input("enter spring constants"))
m=eval(input("enter masses"))
c=eval(input("enter damping constants"))
matm=np.array([[m[0],0],[0,m[1]]])
matk=np.array([[k[0]+k[1],-k[1]],[-k[1],k[1]+k[2]]])
matc=np.array([[c[0]+c[1],-c[1]],[-c[1],c[1]+c[2]]])
def ode(x,Y):
    y_der=xd.grad.jacobian(Y,x,j=0)
    y_dder=xd.grad.hessian(Y,x,j=0)
    return xd.tf.matmul(matm,y_dder)+xd.tf.matmul(matk,Y)+xd.tf.matmul(matc,y_der)
y10=float(input("y1 at t=0"))
y20=float(input("y2 at t=0"))
y1d0=float(input("derivative of y1 at t=0"))
y2d0=float(input("derivative of y2 at t=0"))
w1,w2=[((((k[0]+k[1])*m[1])+((k[2]+k[1])*m[0]))/2)+np.sqrt((((k[0]+k[1])*m[1])+((k[2]+k[1])*m[0]))**(2)-4*(((k[0]+k[1])*(k[1]+k[2])-k[1]**2)/m[0]*m[1])),((((k[0]+k[1])*m[1])+((k[2]+k[1])*m[0]))/2)-np.sqrt((((k[0]+k[1])*m[1])+((k[2]+k[1])*m[0]))**(2)-4*(((k[0]+k[1])*(k[1]+k[2])-k[1]**2)/m[0]*m[1]))]
r1,r2=[k[1]/(-m[1]*w1+(k[1]+k[2])),k[1]/(-m[1]*w2+(k[1]+k[2]))]
def soln(t):
    x11=(((r2*y10-y20)**(2)+((-r2*y1d0+y2d0)**2)/w1)**(1/2))/(r2-r1)
    return[x11,]