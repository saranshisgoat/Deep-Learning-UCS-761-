import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate):
        # Initialize parameters based on layers
        self.learning_rate = learning_rate
        self.layers = layers
        self.weights = []
        self.biases = []

        # Initialize weights and biases
        for i in range(len(layers) - 1):
            weight = np.random.randn(layers[i], layers[i + 1])
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return x * (1 - x)

    def forward(self, X):
        """Forward pass."""
        self.a = [X]  # List to hold activations
        for i in range(len(self.weights)):
            z = np.dot(self.a[i], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, X, y):
        """Backward pass."""
        output_error = self.a[-1] - y  # Error in output
        deltas = [output_error * self.sigmoid_derivative(self.a[-1])]

        # Backpropagate the error
        for i in range(len(self.weights) - 1, 0, -1):
            hidden_error = np.dot(deltas[-1], self.weights[i].T)
            hidden_delta = hidden_error * self.sigmoid_derivative(self.a[i])
            deltas.append(hidden_delta)

        deltas.reverse()  # Reverse the deltas to match weights

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(self.a[i].T, deltas[i])
            self.biases[i] -= self.learning_rate * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs):
        """Train the neural network."""
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, y)
            # Optionally, print the loss every so often
            if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
                loss = np.mean(np.square(y - self.a[-1]))
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.6f}")

    def predict(self, X):
        """Make predictions."""
        return self.forward(X)

# Sample dataset (XOR problem)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize and train the neural network
layers = [2, 2, 1]  # Define the architecture
learning_rate = 0.1
epochs = 100_000

nn = NeuralNetwork(layers, learning_rate)
nn.train(X, y, epochs)

# Make predictions
print("\nPredictions after training:")
for x_input, y_true in zip(X, y):
    predicted = nn.predict(x_input.reshape(1, -1))
    print(f"Input: {x_input} - Predicted: {predicted[0][0]:.4f} - True: {y_true[0]}")