import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Custom function to split the dataset into training and test sets
def split_train_test(X, y, test_ratio=0.2, seed=None):
    if seed is not None:
        np.random.seed(seed)

    total_samples = len(X)
    test_samples = int(total_samples * test_ratio)
    shuffled_indices = np.random.permutation(total_samples)
    
    test_set_indices = shuffled_indices[:test_samples]
    train_set_indices = shuffled_indices[test_samples:]

    X_train, X_test = X[train_set_indices], X[test_set_indices]
    y_train, y_test = y[train_set_indices], y[test_set_indices]

    return X_train, X_test, y_train, y_test

# Standardize the features
def normalize(X, epsilon=1e-8):
    mean_value = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0) + epsilon
    return (X - mean_value) / std_dev, mean_value, std_dev

class SimpleLinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.losses = []

    # Fit the model to the data
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # Make predictions
            predictions = np.dot(X, self.weights) + self.bias

            # Compute the loss (mean squared error)
            loss = np.mean((predictions - y) ** 2)
            self.losses.append(loss)

            # Display progress at intervals
            if epoch == 0 or (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

            # Compute gradients
            dW = (1 / num_samples) * np.dot(X.T, (predictions - y))
            dB = (1 / num_samples) * np.sum(predictions - y)

            # Gradient clipping to avoid large updates
            dW = np.clip(dW, -1, 1)
            dB = np.clip(dB, -1, 1)

            # Update learning rate with decay
            current_lr = self.learning_rate / (1 + 0.01 * epoch)

            # Update weights and bias
            self.weights -= current_lr * dW
            self.bias -= current_lr * dB

    # Predict values for given inputs
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Function to calculate mean squared error
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example of using Linear Regression
print("Linear Regression Example:")

# Load dataset
dataset = pd.read_csv('data_for_lr.csv')
X = dataset['x'].values.reshape(-1, 1)
y = dataset['y'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_train_test(X, y, test_ratio=0.2, seed=42)

# Standardize features
X_train_scaled, train_mean, train_std = normalize(X_train)
X_test_scaled = (X_test - train_mean) / train_std

# Initialize and train the linear regression model
model = SimpleLinearRegression(learning_rate=0.1, epochs=10000)
model.fit(X_train_scaled, y_train)

# Predict on test data
y_predictions = model.predict(X_test_scaled)

# Evaluate performance
mse_value = mse_loss(y_test, y_predictions)
print(f"Mean Squared Error: {mse_value:.4f}")

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_predictions, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.show()

# Plot the loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(model.losses) + 1), model.losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Over Epochs')
plt.yscale('log')  # Logarithmic scale for loss visualization
plt.show()
