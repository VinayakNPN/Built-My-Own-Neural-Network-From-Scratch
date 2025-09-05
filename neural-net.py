import numpy as np

def SigMoid(x):
    #*Activation Function formula: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x)) 


def deriv_sigmoid(x):
    fx = SigMoid(x)
    return fx * (1 - fx)

def mse_loss(y_true,y_pred):
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:
    def __init__(self):
        #Weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        #Biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    
    def feedforward(self, x):
        h1 = SigMoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = SigMoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = SigMoid(self.w5 * h1 + selfw6 * h2 + self.b3)
        return o1
    
    def train(self, data, all_y_trues):
        learn_rate = 0.1
        epochs = 1000

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = SigMoid(sum_h1)

                sum_h2 = self.w2 * x[0] + self.w2 * x[1] + self.b2
                h2 = SigMoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = SigMoid(sum_o1)
                y_pred = o1

                #Calculating partial derivatives
                d_l_d_ypred = -2 * (y_true - y_pred)

                #Neuron o1

                




class Neuron:
    def __init__(self,weights,bias):
        self.weights = weights
        self.bias = bias
    def feedforward(self,inputs):
        total = np.dot(self.weights,inputs) + self.bias
        return SigMoid(total)
    
weights = np.array([0,1]) #!w1 = 0 , w2 = 1
bias = 4
n = Neuron(weights,bias)

x = np.array([2,3]) #* x1 = 2 , x2 = 3
print(n.feedforward(x)) #?0.9990889488055994


    
network = OurNeuralNetwork()
x = np.array([2,3])
print(network.feedforward(x))




y_true = np.array([1,0,0,1,0,0,1])
y_pred = np.array([0,1,1,0,1,1,0])
print(mse_loss(y_true,y_pred))