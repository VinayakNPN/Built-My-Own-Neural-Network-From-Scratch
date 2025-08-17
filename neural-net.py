import numpy as np

def SigMoid(x):
    #*Activation Function formula: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x)) 

class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias
    def FeedForward(self,inputs):
        total = np.dot(self.weights,inputs) + self.bias
        return SigMoid(total)
    
