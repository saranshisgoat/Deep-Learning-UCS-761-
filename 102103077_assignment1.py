import numpy as np 

def mp_neuron(inputs, weights, threshold):
    total = np.dot(inputs, weights)
    return 1 if total >= threshold else 0

def AND_gate(x1, x2):
    return mp_neuron([x1, x2], [1, 1], 2)

def OR_gate(x1, x2):
    return mp_neuron([x1, x2], [1, 1], 1)

def NOR_gate(x1, x2):
    return mp_neuron([x1, x2], [-1, -1], 0)

def NOT_gate(x):
    return mp_neuron([x], [-1], 0)

def sigmoid_function(x):
    return 1.0 / (1 + np.exp(-x))

def sigmoid_gradient(x):
    return x * (1 - x)

class SimplePerceptron: 
    def __init__(self, input_count, learning_rate=0.1, training_cycles=1000): 
        self.weights = np.random.rand(input_count)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.training_cycles = training_cycles
    
    def predict(self, inputs):
        total = np.dot(inputs, self.weights) + self.bias
        return sigmoid_function(total)
    
    def train(self, training_data, labels):
        for _ in range(self.training_cycles):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                adjustment = error * sigmoid_gradient(prediction)
                self.weights += self.learning_rate * adjustment * inputs
                self.bias += self.learning_rate * adjustment

# Testing the logic gates
print("Testing AND Gate:")
print(f"0 AND 0 = {AND_gate(0, 0)}")
print(f"0 AND 1 = {AND_gate(0, 1)}")
print(f"1 AND 0 = {AND_gate(1, 0)}")
print(f"1 AND 1 = {AND_gate(1, 1)}")

print("\nTesting OR Gate:")
print(f"0 OR 0 = {OR_gate(0, 0)}")
print(f"0 OR 1 = {OR_gate(0, 1)}")
print(f"1 OR 0 = {OR_gate(1, 0)}")
print(f"1 OR 1 = {OR_gate(1, 1)}")

print("\nTesting NOR Gate:")
print(f"0 NOR 0 = {NOR_gate(0, 0)}")
print(f"0 NOR 1 = {NOR_gate(0, 1)}")
print(f"1 NOR 0 = {NOR_gate(1, 0)}")
print(f"1 NOR 1 = {NOR_gate(1, 1)}")

print("\nTesting NOT Gate:")
print(f"NOT 0 = {NOT_gate(0)}")
print(f"NOT 1 = {NOT_gate(1)}")

# Training the AND gate with 2-bit inputs
print("Training AND Gate with 2-bit inputs:")
perceptron_2bit = SimplePerceptron(2, training_cycles=100_000)
inputs_2bit = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels_2bit = np.array([0, 0, 0, 1])
perceptron_2bit.train(inputs_2bit, labels_2bit)

for inputs in inputs_2bit:
    prediction = perceptron_2bit.predict(inputs)
    print(f"{inputs[0]} AND {inputs[1]} = {prediction:.4f} (rounded: {round(prediction)})")

# Training the AND gate with 3-bit inputs
print("\nTraining AND Gate with 3-bit inputs:")
perceptron_3bit = SimplePerceptron(3, training_cycles=100_000)
inputs_3bit = np.array([
    [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
])
labels_3bit = np.array([0, 0, 0, 0, 0, 0, 0, 1])
perceptron_3bit.train(inputs_3bit, labels_3bit)

for inputs in inputs_3bit:
    prediction = perceptron_3bit.predict(inputs)
    print(f"{inputs[0]} AND {inputs[1]} AND {inputs[2]} = {prediction:.4f} (rounded: {round(prediction)})")
