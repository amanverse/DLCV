import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))  # Ensuring numerical stability

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, x, y, learning_rate):
        m = x.shape[0]
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, x, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(x)
            loss = self.cross_entropy_loss(y, y_pred)
            self.backward(x, y, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, x):
        y_pred = self.forward(x)
        return np.argmax(y_pred, axis=1)

# Initialize dataset
np.random.seed(42)
X = np.random.rand(500, 3)  # 500 samples, 3 input features
Y = np.zeros((500, 2))
Y[np.random.choice(500, 250, replace=False), 1] = 1  binary classification
Y[:, 0] = 1 - Y[:, 1]  #one-hot encoding

# Initialize and train neural network
nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=2)
nn.train(X, Y, epochs=1000, learning_rate=0.01)

# Test prediction
predictions = nn.predict(X)
accuracy = np.mean(np.argmax(Y, axis=1) == predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")
