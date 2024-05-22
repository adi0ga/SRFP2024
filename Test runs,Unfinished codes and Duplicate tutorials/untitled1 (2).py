import tensorflow as tf
import timeit
from datetime import datetime

def funct(x,y,b):
    x=tf.matmul(x,y)
    x=x+b
    return x
dunct=tf.function(funct)
x1=tf.constant([[1.0,2.0]])
y1=tf.constant([[2.0],[3.0]])
b1=tf.constant(4.0)
orig=funct(x1,y1,b1).numpy()
functtf=dunct(x1,y1,b1).numpy()
assert(orig==functtf)
