import numpy as np

class Perceptron:
    def __init__(self, tam):
        self.weights = np.zeros(tam + 1)

    def predict(self, entradas):
        summa = np.dot(entradas, self.weights[1:]) + self.weights[0]
        return 1 if summa > 0 else 0

    def train(self, entradas, l, epochs=10, apr=0.1):
        for _ in range(epochs):
            for x, y in zip(entradas, l):
                pred = self.predict(x)
                self.weights[1:] += apr * (y - pred) * x
                self.weights[0] += apr * (y - pred)

entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
l = np.array([0, 0, 0, 1])

perceptron = Perceptron(tam=2)
perceptron.train(entradas, l, epochs=10, apr=0.1)

test_inputs = np.array([[1,0]])
for test_input in test_inputs:
    pred = perceptron.predict(test_input)
    print(f"{test_input} : {pred}")