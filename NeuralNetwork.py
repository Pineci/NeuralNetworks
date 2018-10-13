import numpy as np
from enum import Enum



def ReLUElement(x):
    if x < 0:
        return 0
    else:
        return x


def ReLUDerivativeElement(x):
    if x < 0:
        return 0
    else:
        return 1


ReLU = np.vectorize(ReLUElement)
ReLUDerivative = np.vectorize(ReLUDerivativeElement)

class Activation(Enum):
    SIGMOID = 1
    TANH = 2
    RELU = 3


class NeuralNetwork():

    def __init__(self, layers, activation):
        self.layers = layers
        self.activation = activation
        self.weights = []
        self.bias = []
        self.xData = []
        self.yData = []
        self.weightDecay = 0

    def initializeWeights(self, minmax = [-5, 5]):
        a = minmax[0]
        b = minmax[1]
        for i in range(len(self.layers)-1):
            rows = self.layers[i+1]
            cols = self.layers[i]
            self.weights.append(np.matrix((b-a) * np.random.rand(rows, cols) + a))
            self.bias.append(np.matrix((b-a) * np.random.rand(rows, 1) + a))

    def trainingData(self, xData, yData):
        self.xData = xData
        self.yData = yData

    def setWeightDecay(self, weightDecay):
        self.weightDecay = weightDecay

    def activationFunction(self, v):
        if self.activation == Activation.SIGMOID:
            return 1/(1+np.exp(-1*v))
        elif self.activation == Activation.TANH:
            return np.tanh(v)
        elif self.activation == Activation.RELU:
            return ReLU(v)

    def activationFunctionDerivative(self, v):
        if self.activation == Activation.SIGMOID:
            s = self.activationFunction(v)
            return np.multiply(s,1-s)
        elif self.activation == Activation.TANH:
            return 1 - self.activationFunction(v) ** 2
        elif self.activation == Activation.RELU:
            return ReLUDerivative(v)

    def layersAndActivations(self, x):
        layers = []
        activations = [x]
        a = x
        for i in range(len(self.weights)):
            z = np.matmul(self.weights[i], a) + self.bias[i]
            a = self.activationFunction(z)
            layers.append(z)
            activations.append(a)
        return [layers, activations]

    def hypothesis(self, x):
        activation = x
        for i in range(len(self.weights)):
            activation = self.activationFunction(np.matmul(self.weights[i], activation) + self.bias[i])
        return activation

    def costFunction(self):
        hyp = self.hypothesis(self.xData)
        m = self.xData.shape[1]
        sqError = 0.5 * 1/m * np.square(hyp - self.yData).sum()
        decay = self.weightDecay * sum([np.square(w).sum() for w in self.weights])
        return sqError + decay

    def backPopagation(self):
        m = self.xData.shape[1]
        za = self.layersAndActivations(self.xData)
        layers = za[0]
        activations = za[1]
        deltas = [-1*np.multiply(self.yData - activations[-1], self.activationFunctionDerivative(layers[-1]))]
        for i in range(len(self.weights)-1):
            deltas.append(np.multiply(np.matmul(np.transpose(self.weights[len(self.weights)-1-i]), deltas[i]), self.activationFunctionDerivative(layers[len(layers)-1-i])))
        deltas = deltas[::-1]
        wGrad = []
        bGrad = []
        for i in range(len(self.weights)):
            wGrad.append(1/m * np.matmul(deltas[i], np.transpose(activations[i])) + self.weightDecay * self.weights[i])
            bGrad.append(1/m * deltas[i].sum(axis=1))
        return [wGrad, bGrad]

    def shapes(self):
        weightShapes = []
        biasShapes = []
        for i in range(len(self.weights)):
            weightShapes.append(self.weights[i].shape)
            biasShapes.append(self.bias[i].shape)
        return [weightShapes, biasShapes]

    def vectorizeWeights(self, weights, bias):
        parameters = np.array([])
        for i in range(len(weights)):
            #parameters = np.asarray(weights[i]).flatten()
            parameters = np.concatenate((parameters, np.asarray(weights[i]).flatten()))
        for i in range(len(bias)):
            parameters = np.concatenate((parameters, np.asarray(bias[i]).flatten()))
        return parameters

    def unvectorizeWeights(self, vector):
        weightShapes, biasShapes = self.shapes()
        weights = []
        bias = []
        index = 0
        for i in range(len(weightShapes)):
            size = weightShapes[i][0] * weightShapes[i][1]
            weights.append(np.asmatrix(np.reshape(vector[index:index+size], weightShapes[i])))
            index += size
        for i in range(len(biasShapes)):
            size = biasShapes[i][0] * biasShapes[i][1]
            bias.append(np.asmatrix(np.reshape(vector[index:index + size], biasShapes[i])))
            index += size
        return [weights, bias]

    def gradientDescent(self, iter):
        alpha = 0.5
        prevV = self.vectorizeWeights(self.weights, self.bias)
        [wGrad, bGrad] = self.backPopagation()
        prevGrad = self.vectorizeWeights(wGrad, bGrad)
        newV = prevV - alpha * prevGrad
        [newWeights, newBias]
        for j in range(iter):
            alpha = (newV - prevV).T




n = NeuralNetwork([1, 2, 1], Activation.SIGMOID)
n.initializeWeights()
n.setWeightDecay(0)
xSinData = np.asmatrix(np.random.rand(100) * 100)
ySinData = np.abs(np.sin(xSinData))
xSinTestData = np.asmatrix(np.random.rand(100) * 100)
ySinTestData = np.abs(np.sin(xSinTestData))
xData = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
yData = np.matrix([[1, 0, 1]])
n.trainingData(xSinData, ySinData)
#print(n.costFunction())
n.gradientDescent(5000)
vec = n.vectorizeWeights(n.weights, n.bias)
print(n.unvectorizeWeights(vec))
#print(n.weights)
