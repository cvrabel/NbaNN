import numpy as np

class Neural_Network(object):
    def __init__(self, Lambda, inputSize, hiddenSize, outputSize):

        self.inputLayerSize = inputSize
        self.outputLayerSize = outputSize
        self.hiddenLayerSize = hiddenSize
  
        #Weights (parameters)
        #W1 for weights from input to hidden (initialized as random)
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        #W2 for weights from hidden to output (initialized as random)
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        
        #Regularization Parameter:
        self.Lambda = Lambda
        
    def propogation(self, inputX):
        #Propogate inputs though network

        self.secondLayer = np.dot(inputX, self.W1)
        #Apply activation function
        self.sigmoidSecond = 1/(1+np.exp(self.secondLayer))

        self.thirdLayer = np.dot(self.sigmoidSecond, self.W2)
        #Apply activation function
        yHat = 1/(1+np.exp(self.thirdLayer))
        return yHat
        
    
    def sigmoidPrime(self,z):
        #Gradient of sigmoid (get the derivative)
        return np.exp(-z)/( (1+np.exp(-z))**2 )
    
    def costFunction(self, X, y):
        #Compute how much error for given X,y, use weights already stored in class.
        #Need to minimize error(cost)
        self.yHat = self.propogation(X)

        '''Square the errors so cannot get stuck in local min when performing 
        gradient descent.  Will make the graph convex.'''
        error = 0.5*np.sum( (y-self.yHat)**2 )/X.shape[0] + \
                (self.Lambda/2)*( np.sum(self.W2**2)+np.sum(self.W2**2) )
        return error
        
    def costFunctionPrime(self, X, y):
        # print('cost')
        '''Compute partial derivative with respect to W and W2 for a given X and y:
        Gradient descent method: check error at steps and stop when error stops getting smaller.'''
        
        self.yHat = self.propogation(X)
        errorThirdLayer = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.thirdLayer))
        partialDerivW2 = np.dot(self.sigmoidSecond.T, errorThirdLayer)/X.shape[0] + self.Lambda*self.W2

        errorSecondLayer = np.dot(errorThirdLayer, self.W2.T)*self.sigmoidPrime(self.secondLayer)
        partialDerivW1 = np.dot(X.T, errorSecondLayer)/X.shape[0] + self.Lambda*self.W1
        
        return partialDerivW1, partialDerivW2
    
    #Helper functions for interacting with other methods/classes
    def getParams(self):
        #Get W1 and W2 Rolled into vector using ravel
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    
    def setParams(self, params):
        #Set W1 and W2 using single parameter vector:

        W1_final = self.hiddenLayerSize*self.inputLayerSize #Number of weights in W1
        self.W1 = np.reshape(params[0:W1_final], (self.inputLayerSize, self.hiddenLayerSize))

        W2_final = W1_final + self.hiddenLayerSize*self.outputLayerSize #Number of weights in W2
        self.W2 = np.reshape(params[W1_final:W2_final], (self.hiddenLayerSize, self.outputLayerSize))
        
    def computeGradients(self, X, y):
        #Get the two gradients from the cost function 
        #then return them vectorized
        partialDerivW1, partialDerivW2 = self.costFunctionPrime(X, y)
        return np.concatenate((partialDerivW1.ravel(), partialDerivW2.ravel()))

        
## ----------------------- Part 6 ---------------------------- ##
from scipy import optimize


##Need to modify trainer class a bit to check testing error during training:
class trainer(object):
    def __init__(self, NeuralNetwork):
        #Make Local reference to network:
        self.NN = NeuralNetwork
        
    def callbackF(self, params):
        self.NN.setParams(params)
        self.Costs.append(self.NN.costFunction(self.X, self.y))
        self.testCosts.append(self.NN.costFunction(self.testX, self.testY))
        
    def costFunctionWrapper(self, params, X, y):
        #Needed for the scipy optimize method
        self.NN.setParams(params)
        error = self.NN.costFunction(X, y)
        gradient = self.NN.computeGradients(X,y)
        
        return error, gradient
        
    def train(self, trainX, trainY, testX, testY):
        #Make an internal variable for the callback function:
        self.X = trainX
        self.y = trainY
        
        self.testX = testX
        self.testY = testY

        #Make empty list to store training costs:
        self.Costs = []
        self.testCosts = []
        
        #Use the scipy optimize function
        #Use BFGS to estimate the curvature and useful for batch descent
        parameters = self.NN.getParams()
        options = {'maxiter': 600, 'disp' : False}

        #First parameter must be a function that accepts vector of parameters and in/out data
        #and returns the costs and gradients 
        _res = optimize.minimize(self.costFunctionWrapper, parameters, jac=True, method='BFGS', \
                                 args=(trainX, trainY), options=options, callback=self.callbackF)

        self.NN.setParams(_res.x)
        self.optimizationResults = _res


















