import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def train_test_split(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

def standardize(X, epsilon=1e-8):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + epsilon
    return (X - mean) / std, mean, std

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_reg=0.01):
        """
        Initializes the Linear Regression model.
        
        Parameters:
        - learning_rate: The step size for gradient descent.
        - n_iterations: Number of iterations for training.
        - regularization: Type of regularization ('l1', 'l2', or None).
        - lambda_reg: Regularization strength.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Compute loss
            loss = np.mean((y_predicted - y) ** 2)
            if self.regularization == 'l2':
                loss += self.lambda_reg * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += self.lambda_reg * np.sum(np.abs(self.weights))
            self.loss_history.append(loss)
            
            if i == 0 or (i + 1) % 1000 == 0 or i == self.n_iterations -1:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}")
            
            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)
            
            # Apply regularization to gradients
            if self.regularization == 'l2':
                dw += 2 * self.lambda_reg * self.weights
            elif self.regularization == 'l1':
                dw += self.lambda_reg * np.sign(self.weights)
            
            # Gradient clipping (optional)
            dw = np.clip(dw, -1, 1)
            db = np.clip(db, -1, 1)
            
            # Learning rate decay (optional)
            current_lr = self.learning_rate / (1 + 0.01 * i)
            
            # Update parameters
            self.weights -= current_lr * dw
            self.bias -= current_lr * db
    
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization=None, lambda_reg=0.01):
        """
        Initializes the Logistic Regression model.
        
        Parameters:
        - learning_rate: The step size for gradient descent.
        - n_iterations: Number of iterations for training.
        - regularization: Type of regularization ('l1', 'l2', or None).
        - lambda_reg: Regularization strength.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)
            
            # To prevent log(0), clip y_predicted
            y_predicted = np.clip(y_predicted, 1e-15, 1 - 1e-15)
            
            # Compute loss
            loss = -np.mean(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
            if self.regularization == 'l2':
                loss += self.lambda_reg * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += self.lambda_reg * np.sum(np.abs(self.weights))
            self.loss_history.append(loss)
            
            if i ==0 or (i + 1) % 1000 == 0 or i == self.n_iterations -1:
                print(f"Iteration {i+1}/{self.n_iterations}, Loss: {loss:.4f}")
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Apply regularization to gradients
            if self.regularization == 'l2':
                dw += 2 * self.lambda_reg * self.weights
            elif self.regularization == 'l1':
                dw += self.lambda_reg * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        return (y_predicted > 0.5).astype(int)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Linear Regression Example with Regularization
print("Linear Regression Example with Regularization:")

# Generate synthetic data for Linear Regression
np.random.seed(42)
n_samples = 100
X = 2 * np.random.rand(n_samples, 1)
true_weights = 3.5
true_bias = 1.2
y = true_weights * X.squeeze() + true_bias + np.random.randn(n_samples) * 0.5  # Adding some noise

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
X_train_scaled, mean, std = standardize(X_train)
X_test_scaled = (X_test - mean) / std

# Create and train the linear regression model with L2 regularization
lr_model_l2 = LinearRegression(learning_rate=0.1, n_iterations=10000, regularization='l2', lambda_reg=0.1)
lr_model_l2.fit(X_train_scaled, y_train)

# Make predictions
y_pred_l2 = lr_model_l2.predict(X_test_scaled)

# Evaluate the model
mse_l2 = mean_squared_error(y_test, y_pred_l2)
r2_l2 = r2_score(y_test, y_pred_l2)

print(f"L2 Regularization - Mean Squared Error: {mse_l2:.4f}")
print(f"L2 Regularization - R-squared Score: {r2_l2:.4f}")

# Create and train the linear regression model with L1 regularization
lr_model_l1 = LinearRegression(learning_rate=0.1, n_iterations=10000, regularization='l1', lambda_reg=0.1)
lr_model_l1.fit(X_train_scaled, y_train)

# Make predictions
y_pred_l1 = lr_model_l1.predict(X_test_scaled)

# Evaluate the model
mse_l1 = mean_squared_error(y_test, y_pred_l1)
r2_l1 = r2_score(y_test, y_pred_l1)

print(f"L1 Regularization - Mean Squared Error: {mse_l1:.4f}")
print(f"L1 Regularization - R-squared Score: {r2_l1:.4f}")

# Plot the results
plt.figure(figsize=(14, 6))

# L2 Regularization Plot
plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred_l2, color='red', label='Predicted (L2)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with L2 Regularization: Actual vs Predicted')
plt.legend()

# L1 Regularization Plot
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred_l1, color='green', label='Predicted (L1)')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with L1 Regularization: Actual vs Predicted')
plt.legend()

plt.tight_layout()
plt.show()

# Plot loss curves
plt.figure(figsize=(14, 6))

# L2 Loss Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, len(lr_model_l2.loss_history) + 1), lr_model_l2.loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Linear Regression with L2 Regularization: Loss Curve')
plt.yscale('log')  # Use log scale for better visualization

# L1 Loss Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, len(lr_model_l1.loss_history) + 1), lr_model_l1.loss_history, color='green')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Linear Regression with L1 Regularization: Loss Curve')
plt.yscale('log')  # Use log scale for better visualization

plt.tight_layout()
plt.show()

# Logistic Regression Example with Regularization
print("\nLogistic Regression Example with Regularization:")

# Generate sample data
np.random.seed(0)
X_log = np.random.randn(100, 2)
y_log = (X_log[:, 0] + X_log[:, 1] > 0).astype(int)

# Split the data
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

# Standardize the data
X_train_scaled_log, mean_log, std_log = standardize(X_train_log)
X_test_scaled_log = (X_test_log - mean_log) / std_log

# Create and train the logistic regression model with L2 regularization
log_model_l2 = LogisticRegression(learning_rate=0.1, n_iterations=10000, regularization='l2', lambda_reg=0.1)
log_model_l2.fit(X_train_scaled_log, y_train_log)

# Make predictions
y_pred_log_l2 = log_model_l2.predict(X_test_scaled_log)

# Evaluate the model
accuracy_l2 = accuracy_score(y_test_log, y_pred_log_l2)
print(f"L2 Regularization - Accuracy: {accuracy_l2:.4f}")

# Create and train the logistic regression model with L1 regularization
log_model_l1 = LogisticRegression(learning_rate=0.1, n_iterations=10000, regularization='l1', lambda_reg=0.1)
log_model_l1.fit(X_train_scaled_log, y_train_log)

# Make predictions
y_pred_log_l1 = log_model_l1.predict(X_test_scaled_log)

# Evaluate the model
accuracy_l1 = accuracy_score(y_test_log, y_pred_log_l1)
print(f"L1 Regularization - Accuracy: {accuracy_l1:.4f}")

# Plot the decision boundaries
def plot_decision_boundary(model, X, y, title):
    plt.figure(figsize=(8,6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label='Class 0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label='Class 1')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.legend()
    
    # Plot decision boundary
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),
                           np.arange(x2_min, x2_max, 0.1))
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, levels=[-0.1, 0.5, 1.1], colors=['lightblue','lightcoral'])
    plt.show()

# Plot decision boundaries for L2 and L1
plot_decision_boundary(log_model_l2, X_test_log, y_test_log, 'Logistic Regression with L2 Regularization: Decision Boundary')
plot_decision_boundary(log_model_l1, X_test_log, y_test_log, 'Logistic Regression with L1 Regularization: Decision Boundary')

# Plot loss curves
plt.figure(figsize=(14, 6))

# L2 Loss Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, len(log_model_l2.loss_history) + 1), log_model_l2.loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Logistic Regression with L2 Regularization: Loss Curve')
plt.yscale('log')  # Use log scale for better visualization

# L1 Loss Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, len(log_model_l1.loss_history) + 1), log_model_l1.loss_history, color='green')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Logistic Regression with L1 Regularization: Loss Curve')
plt.yscale('log')  # Use log scale for better visualization

plt.tight_layout()
plt.show()