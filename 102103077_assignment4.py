import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Plotting Sigmoid
x = np.linspace(-10, 10, 400)
y = sigmoid(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y, label='Sigmoid', color='blue')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()


def tanh(x):
    return np.tanh(x)

# Plotting Tanh
y_tanh = tanh(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y_tanh, label='Tanh', color='green')
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()

def relu(x):
    return np.maximum(0, x)

# Plotting ReLU
y_relu = relu(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y_relu, label='ReLU', color='red')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()



def leaky_relu(x, alpha=0.01):
    return np.where(x >= 0, x, alpha * x)

# Plotting Leaky ReLU
y_leaky_relu = leaky_relu(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y_leaky_relu, label='Leaky ReLU', color='orange')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()


def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

# Plotting ELU
y_elu = elu(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y_elu, label='ELU', color='purple')
plt.title('ELU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()


def softmax(x):
    e_x = np.exp(x - np.max(x))  # For numerical stability
    return e_x / e_x.sum()

# Example input vectors
vectors = [
    np.array([2.0, 1.0, 0.1]),
    np.array([1.0, 2.0, 3.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([3.0, 1.0, 0.2])
]

# Applying Softmax and plotting
for i, vec in enumerate(vectors, 1):
    softmax_output = softmax(vec)
    plt.bar(range(len(vec)), softmax_output, tick_label=[f'x{j}' for j in range(1, len(vec)+1)])
    plt.ylim(0, 1)
    plt.ylabel('Probability')
    plt.title(f'Softmax Output for Vector {i}')
    plt.show()


def softplus(x):
    return np.log(1 + np.exp(x))

# Plotting Softplus
y_softplus = softplus(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y_softplus, label='Softplus', color='brown')
plt.title('Softplus Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()


def binary_step(x):
    return np.where(x >= 0, 1, 0)

# Plotting Binary Step
y_binary_step = binary_step(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y_binary_step, label='Binary Step', color='cyan')
plt.title('Binary Step Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.ylim(-0.1, 1.1)
plt.grid(True)
plt.legend()
plt.show()




def piecewise_linear(x):
    return np.piecewise(x, 
                        [x < -1, (x >= -1) & (x <= 1), x > 1],
                        [lambda x: 0.5*x + 1, 
                         lambda x: x, 
                         lambda x: 0.5*x + 1])

# Plotting Piecewise Linear
y_piecewise = piecewise_linear(x)

plt.figure(figsize=(8, 4))
plt.plot(x, y_piecewise, label='Piecewise Linear', color='magenta')
plt.title('Piecewise Linear Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.grid(True)
plt.legend()
plt.show()



# Sample input data
test_inputs = np.linspace(-3, 3, 7)  # [-3, -2, -1, 0, 1, 2, 3]
print("Test Inputs:", test_inputs)

# Applying activation functions
sigmoid_outputs = sigmoid(test_inputs)
tanh_outputs = tanh(test_inputs)
relu_outputs = relu(test_inputs)
leaky_relu_outputs = leaky_relu(test_inputs)
elu_outputs = elu(test_inputs)
softplus_outputs = softplus(test_inputs)
binary_step_outputs = binary_step(test_inputs)
piecewise_linear_outputs = piecewise_linear(test_inputs)

# For Softmax, we'll consider the test_inputs as a vector
softmax_output = softmax(test_inputs)

# Displaying Results
import pandas as pd

data = {
    'Input': test_inputs,
    'Sigmoid': sigmoid_outputs,
    'Tanh': tanh_outputs,
    'ReLU': relu_outputs,
    'Leaky ReLU': leaky_relu_outputs,
    'ELU': elu_outputs,
    'Softplus': softplus_outputs,
    'Binary Step': binary_step_outputs,
    'Piecewise Linear': piecewise_linear_outputs
}

df = pd.DataFrame(data)
print(df)

# Softmax Output
print("\nSoftmax Output for test inputs:", softmax_output)