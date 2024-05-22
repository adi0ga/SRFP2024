import torch
import torch.nn as nn
import torch.optim as optim
from FNN import FeedforwardNeuralNetwork
from Dif_op import compute_derivatives
import numpy as np

class PhysicsInformedNN:
    def __init__(self, model, m, k, c, x0, v0):
        self.model = model
        self.m = m
        self.k = k
        self.x0 = x0
        self.v0 = v0
        self.c = c

    def forward(self, t):
        return self.model(t)

    def loss(self, t):
        # Compute model predictions and derivatives
        x, x_t, x_tt = compute_derivatives(self.model, t)

        # Differential equation loss: mx'' + kx should be close to 0
        f = self.m * x_tt + self.k * x + self.c*x_t
        eq_loss = torch.mean(f**2)

        # Boundary conditions
        ic_loss_x = (x[0] - self.x0)**2  # Initial displacement
        ic_loss_v = (x_t[0] - self.v0)**2  # Initial velocity

        # Combine losses
        total_loss = eq_loss + ic_loss_x + ic_loss_v
        return total_loss

    def train(self, t, epochs, optimizer):
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = self.loss(t)
            loss.backward()
            optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item()}')

