from Pre_Process import preprocess
from FNN import FeedforwardNeuralNetwork
from PINN import PhysicsInformedNN
import torch.optim as optim
import torch
import numpy as np
from Input_Info import analytical_solution
import matplotlib.pyplot as plt

# Define the problem parameters
m = 2.0  # Mass
k = 2  # Spring constant
x0 = 0.5  # Initial displacement
v0 = 1 # Initial velocity
c= 4  #damping coefficient
# Define the training configuration
t_start = 0.0
t_end = 10.0
num_points = 1000
epochs = 5000
learning_rate = 0.1

def main():
    # Step 1: Preprocess - Generate time points
    t_tensor = preprocess(t_start, t_end, num_points)

    # Step 2: Set up the model
    model = FeedforwardNeuralNetwork(input_size=1, output_size=1, hidden_layers=[10], activation='tanh')

    # Step 3: Instantiate the PINN
    pinn = PhysicsInformedNN(model, m, k, c, x0, v0)

    # Step 4: Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Step 5: Train the model
    print("Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = pinn.loss(t_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    print('Training completed.')
    # Predictions from the trained model
    x_pred = pinn.forward(t_tensor).detach().numpy().flatten()

    # Analytical solution for comparison
    t_vals = t_tensor.detach().numpy().flatten()
    x_true = analytical_solution(t_vals,m,c,k,x0,v0)

    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, x_pred, label='PINN Solution', linewidth=2)
    plt.plot(t_vals, x_true, 'r--', label='Analytical Solution', linewidth=2)
    plt.xlabel('Time (t)')
    plt.ylabel('Displacement (x)')
    plt.legend()
    plt.title('Comparison of PINN and Analytical Solutions')
    plt.show()

if __name__ == "__main__":
    main()
