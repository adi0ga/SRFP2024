import tensorflow as tf
class SimpleModule(tf.Module):
    def __init__(self,name=None):
        super().__init__(name=name)
        self.var=tf.Variable(5.0,name="train")
        self.nonvar=tf.Variable(5.0,trainable=False,name="do_not_train")
    def __call__(self,x):
        return self.var*x+self.nonvar
simple_module=SimpleModule(name="simple")
print(simple_module(tf.constant(5.0)))

# All trainable variables
print("trainable variables:", simple_module.trainable_variables)
# Every variable
print("all variables:", simple_module.variables)







class Dense(tf.Module):
  def __init__(self, in_features, out_features, name=None):
    super().__init__(name=name)
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)


class SequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

  def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)

# You have made a model!
my_model = SequentialModule(name="the_model")

# Call it, with random results
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))
print("Submodules:", my_model.submodules)

for var in my_model.variables:
  print(var, "\n")
