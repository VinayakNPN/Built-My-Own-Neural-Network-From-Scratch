import numpy as np

def SigMoid(x):
    #*Activation Function formula: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x)) 