import deepxde as xd
import tensorflow as tf
import numpy as np
import keras
class feed_forward(tf.Module):
    def __init__(self,name=None):
        super().__init__(name=name)
    def __call__(self,y,t,param):
        return 0
        
        
        
def ode(y,t,param):
    der=xd.grad.jacobian(y, t)
    doub_der=xd.grad.hessian(y, t)
    return param[2]*doub_der+param[1]*der+param[0]
