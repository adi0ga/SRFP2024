import numpy as np

def analytical_solution(t, m, c, k, x0, v0):
    """
    Calculate the displacement of an underdamped spring-mass-damper system at time t.

    Parameters:
    - t : Time at which to calculate the displacement.
    - m : Mass of the system.
    - c : Damping coefficient.
    - k : Spring constant.
    - x0: Initial displacement.
    - v0: Initial velocity.

    Returns:
    - x(t) : Displacement of the system at time t.
    """
    condition = c / (2 * np.sqrt(k * m))
    if condition > 1:
        # Overdamped case
        lambda_1 = -c / (2 * m) + np.sqrt((c / (2 * m))**2 - k / m)
        lambda_2 = -c / (2 * m) - np.sqrt((c / (2 * m))**2 - k / m)
        # Solving for C1 and C2 using initial conditions
        C1 = (v0 - lambda_2 * x0) / (lambda_1 - lambda_2)
        C2 = x0 - C1
        x_t = C1 * np.exp(lambda_1 * t) + C2 * np.exp(lambda_2 * t)
    else:
        # Underdamped case
        omega_d = np.sqrt(k / m - (c / (2 * m))**2)
        A = x0
        B = (v0 + (c / (2 * m)) * x0) / omega_d
        x_t = np.exp(-c / (2 * m) * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    return x_t

