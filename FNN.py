import torch
import torch.nn as nn

class FeedforwardNeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation):
        super(FeedforwardNeuralNetwork, self).__init__()
        # List of all layer sizes: input, hidden layers, and output
        layers_sizes = [input_size] + hidden_layers + [output_size]
        layers = []

        # Initialize the layers
        for i in range(len(layers_sizes) - 1):
            layers.append(nn.Linear(layers_sizes[i], layers_sizes[i + 1]))
            if i < len(layers_sizes) - 2:  # Avoid activation after the last layer
                if activation == 'tanh':
                    layers.append(nn.Tanh())
                elif activation == 'sigmoid':
                    layers.append(nn.Sigmoid())
                elif activation == 'ReLU':
                    layers.append(nn.ReLU())    
                else:
                    raise ValueError(f"Unsupported activation function: {activation}")

        # Model defined as a sequence of layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass through the network
        return self.model(x)

# Model instantiation
if __name__ == '__main__':
    # Define the neural network parameters
    input_size = 1  # Time t as input
    output_size = 1  # Displacement x(t) as output
    hidden_layers = [50, 50, 50]  # Three hidden layers with 50 neurons each
    activation = 'tanh'  # Tanh activation function

    # Instantiate the neural network model
    model = FeedforwardNeuralNetwork(input_size, output_size, hidden_layers, activation)

    print("Neural Network Model:")
    print(model)
