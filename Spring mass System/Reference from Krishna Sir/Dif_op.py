import torch

def compute_derivatives(model, t):
    """
    Compute the first and second derivatives of the model output with respect to its input.

    Parameters:
    - model: The neural network model predicting x(t).
    - t: The input tensor for which derivatives are computed, representing time points.

    Returns:
    - x: The model output at each input time point.
    - x_t: The first derivative of the model output with respect to time.
    - x_tt: The second derivative of the model output with respect to time.
    """
    # Ensure that t requires gradient
    t.requires_grad_(True)

    # Model prediction for x
    x = model(t)

    # First derivative of x with respect to t
    x_t = torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]

    # Second derivative of x with respect to t
    x_tt = torch.autograd.grad(x_t, t, grad_outputs=torch.ones_like(x_t), create_graph=True)[0]

    return x, x_t, x_tt
