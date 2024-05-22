import deepxde as dde
import numpy as np

# User inputs for the system parameters
m = float(input("Enter the mass: "))
c = float(input("Enter the damping constant: "))
k = float(input("Enter the spring constant: "))
x0 = float(input("Enter the initial condition: "))
v0 = float(input("Enter the initial velocity: "))

# Define the ODE system using closures to capture the constants m, c, k
def get_ode_system(m, c, k):
    def ode_system(t, y):
        dy_dt = dde.grad.jacobian(y, t)
        d2y_dt2 = dde.grad.hessian(y, t)
        return m * d2y_dt2 + c * dy_dt + k * y
    return ode_system

ode = get_ode_system(m, c, k)

# Analytical solution function also using closures
def get_analytical_solution(m, c, k, x0, v0):
    def analytical_sol(t):
        omega_n = np.sqrt(k / m)  # Natural frequency
        zeta = c / (2 * np.sqrt(k * m))  # Damping ratio

        if c == 0:  # Undamped
            return x0 * np.cos(omega_n * t) + (v0 / omega_n) * np.sin(omega_n * t)
        elif c**2 < 4 * m * k:  # Underdamped
            omega_d = omega_n * np.sqrt(1 - zeta**2)
            exp_term = np.exp(-zeta * omega_n * t)
            return exp_term * (x0 * np.cos(omega_d * t) + ((v0 + zeta * omega_n * x0) / omega_d) * np.sin(omega_d * t))
        elif c**2 == 4 * m * k:  # Critically damped
            r = -zeta * omega_n
            return (x0 + (v0 - r * x0) * t) * np.exp(r * t)
        else:  # Overdamped
            r1 = -zeta * omega_n + omega_n * np.sqrt(zeta**2 - 1)
            r2 = -zeta * omega_n - omega_n * np.sqrt(zeta**2 - 1)
            A = (v0 - r2 * x0) / (r1 - r2)
            B = (r1 * x0 - v0) / (r1 - r2)
            return A * np.exp(r1 * t) + B * np.exp(r2 * t)
    return analytical_sol

func = get_analytical_solution(m, c, k, x0, v0)

# Time domain and boundary conditions remain unchanged
geom = dde.geometry.TimeDomain(0, 20)

# Correctly capturing initial conditions in boundary functions using closures
def get_bc_func1(x0):
    def bc_func1(inputs, outputs):
        return outputs - x0
    return bc_func1

def get_bc_func2(v0):
    def bc_func2(inputs, outputs, X):
        return dde.grad.jacobian(outputs, inputs, i=0, j=None) - v0
    return bc_func2

ic1 = dde.icbc.IC(geom, lambda x: x0, lambda _, on_initial: on_initial)
ic2 = dde.icbc.OperatorBC(geom, get_bc_func2(v0), lambda t, on_initial: on_initial and dde.utils.isclose(t[0], 0))

data = dde.data.TimePDE(geom, ode, [ic1, ic2], 50, 2, solution=func, num_test=2000)
layer_size = [1] + [37] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, loss_weights=[1.25, 1, 1])
losshistory, train_state = model.train(iterations=10000)

dde.saveplot(losshistory, train_state, issave=True, isplot=True)