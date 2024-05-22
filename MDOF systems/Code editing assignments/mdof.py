import tensorflow as tf
import numpy as np
n=int(input("No. of dof"))
M=eval(input("enter mass matrix"))
M=np.array(M)
C=eval(input("enter damping matrix"))
C=np.array(C)
K=eval(input("enter sping constant matrix"))
K=np.array(K)
lambda_=float(input("Enter loss weight for PINN loss"))
F=eval(input("enter driving force"))
F=np.array(F)
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(n)  # Output layer for displacement, velocity, and acceleration

    def call(self, t):
        x = self.dense1(t)
        x = self.dense2(x)
        u = self.dense3(x)
        return u
def physics_loss(model, t, M, C, K, F):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        u_pred = model(t)
        u, v, a = u_pred[:, :n], u_pred[:, n:2*n], u_pred[:, 2*n:]
        u_t = tape.gradient(u, t)
        u_tt = tape.gradient(u_t, t)
    loss = tf.reduce_mean(tf.square(M @ u_tt + C @ u_t + K @ u - F))
    return loss

def data_loss(u_pred, u_true):
    return tf.reduce_mean(tf.square(u_pred - u_true))

def total_loss(model, t, u_true, M, C, K, F, lambda_):
    u_pred = model(t)
    l_data = data_loss(u_pred, u_true)
    l_physics = physics_loss(model, t, M, C, K, F)
    return l_data + lambda_ * l_physics
model = PINN()
optimizer = tf.keras.optimizers.Adam()
num_epochs=100000
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        loss = total_loss(model, t_train, u_train, M, C, K, F, lambda_)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')
