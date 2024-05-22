import numpy as np
import torch

def generate_time_points(t_start, t_end, num_points):
    """
    Generate a tensor of time points within the specified range.

    Parameters:
    - t_start: The starting time.
    - t_end: The ending time.
    - num_points: The number of time points to generate.

    Returns:
    - A tensor of time points shaped as (num_points, 1).
    """
    t = np.linspace(t_start, t_end, num_points)
    t_tensor = torch.Tensor(t.reshape(-1, 1))  # Reshape for a single feature
    t_tensor.requires_grad = True  # Enable gradient computation for t
    return t_tensor

def preprocess(t_start=0.0, t_end=50.0, num_points=100):
    """
    Preprocess the data for the PINN model training, including generating time points.

    Parameters:
    - t_start: The starting time for the evaluation.
    - t_end: The ending time for the evaluation.
    - num_points: The number of time points to generate.

    Returns:
    - A tensor of time points.
    """
    # Generate time points
    t_tensor = generate_time_points(t_start, t_end, num_points)
    return t_tensor


