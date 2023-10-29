import array
import os
import numpy as np 
import pandas as pd 
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)

class LogisticRegression:
    def __init__(self, learningRate=0.01, treshold=0.01, epochs=5, inputDimension=2, layerDimension=2, layers=2):

        #Initializing hyperparameters
        self.learningRate = learningRate
        self.treshold = treshold
        self.epochs = epochs
        self.inputDimension = inputDimension
        self.layerDimension = layerDimension
        self.layers = layers
        self.converged = False
        np.random.seed(10) #So "random" weights and bias are initialised likewise between different models.

        #Initializing weights randomly, with specified dimension and number of layers.
        if layers > 2:
            firstLayer = np.random.rand(layerDimension, int(inputDimension*(layerDimension)/layerDimension) )
            self.w = [firstLayer]

            deepLayers = np.random.rand(int((layers-2)), int(((layerDimension**2))/layerDimension) , layerDimension )
            for elem in deepLayers:   
               self.w.append(elem)

            lastLayer = np.random.rand(layerDimension, int((layerDimension)/layerDimension) )
            self.w.append(lastLayer)
        
        else:
            firstLayer = np.random.rand(layerDimension, int(inputDimension*(layerDimension)/layerDimension) )
            lastLayer = np.random.rand(layerDimension, int((layerDimension)/layerDimension) )
            self.w = [firstLayer, lastLayer]

        #Initializing bias randomly
        self.b = []
        for i in range(layers-1):
            self.b.append(np.random.rand(1, layerDimension)) #bias for hidden layer i
        self.b.append(np.random.rand(1,1)) #bias for outputLayer
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats containing 
                m binary 0.0/1.0 labels
        """
        self.converged = False
        for i in range(self.epochs): #2. repeatedly predicting, calculating errorTerms and updating weights and bias until convergence
            print("epoch: ", i)
            predictions = self.predict(X) #Predicting on all examples
            self.stochasticBackpropagation(y, predictions) #Performing weight-optimization
        self.converged = True #So correct value is returned in predict()


    def updateWeightsAndBias(self, errorTerms, yPred):
        for i in range(len(self.w)): #Iterating through each weight-layer, and updating weights in accordance with algorithm in chapter 4.5.2 (ML (Tom M. Mitchell)
            yPredFitted = np.tile(yPred[i], (len(errorTerms[i][0]), 1)) #inputs must be reformatted so they can multiply correctly with their respective weights.
            if i == len(self.w)-1: #Last set of weights cannot be updated likewise as deep layers.
                self.w[i] = np.add(np.transpose(self.w[i]), np.multiply(np.multiply(self.learningRate, errorTerms[i]),  yPredFitted.transpose()).transpose())
                self.w[i] = self.w[i].transpose() 
            else: #Deep layers
                self.w[i] = np.add(self.w[i], np.multiply(np.multiply(self.learningRate, errorTerms[i]), yPredFitted.transpose()).transpose())
            self.b[i] = np.add(self.b[i], np.multiply( self.learningRate, errorTerms[i])) #Bias-weights are updated likewise as weights, only with 1 as constant input.


    def stochasticBackpropagation(self, y, yPred): 
        for i in range(len(yPred)): #Iterating over predictions, and computing outputErrors in accordance with algorithm in chapter 4.5.2 (ML (Tom M. Mitchell) 
            errorTerms = [np.zeros_like(elem) for elem in yPred[0][1:len(yPred[0])]] #creating errorTerms w/zero-values for all nodes in network.
            pred = yPred[i][len(yPred[i])-1] #Output-prediction from output-layer/node
            outputError = np.array((y[i] - pred)*pred*(1-pred)) #Computing errorTerm for output-unit (Assuming only 1 output unit, i.e. binary-classification problem)
            errorTerms[len(errorTerms)-1] = outputError # Setting output-errorTerm
            ones = np.array([1 for i in range(self.layerDimension)]) #Array of 1s w/same dimension as number of nodes in hidden layers.

            for m in range((len(errorTerms)-2), -1, -1): #Iterating backwards in Neural Net
                if m == len(errorTerms)-2:
                    errorTerms[m] = np.multiply(np.transpose(np.dot(self.w[m+1], outputError)),         #Computing errorTerms for nodes in last hidden layer
                                                  np.multiply(yPred[i][m+1], (ones - yPred[i][m+1]))) #(Must be computed otherwise because of different input)    
                else:
                    errorTerms[m] = np.multiply((np.dot(np.transpose(self.w[m+1]), np.transpose(outputError))), #Computing errorTerms for nodes in hidden layers
                                                   np.transpose(np.multiply(yPred[i][m+1], (ones - yPred[i][m+1])))).transpose() #-||-
                outputError = errorTerms[m] #updating outputError to be outputError from layer m to be used on layer m-1. (I.e. concept of backpropagation)
            self.updateWeightsAndBias(errorTerms, yPred[i]) #Updating weights stochastically, i.e. after each prediction-errorTerms

    def predict(self, X):
        allPredictions = []
        for i in range(len(X)): #each example is sent through neural net
            predictions = [X[i]]
            val = [X[i]]
            for m in range(len(self.w)): #Each basically representing a layer in the neural net.
                if m != len(self.w)-1: 
                    val = sigmoid(np.dot(self.w[m], val[0]) + self.b[m]) # hidden-layer-predictions  
                else:
                    val = sigmoid(np.dot((np.transpose(self.w[m])), np.transpose(val))) #output-layer-prediction
                predictions.append(val) #storing predictions (both intermediate and final) from example X[i] in list.
            allPredictions.append(predictions) #storing all predictions from X in list
        
        if self.converged: #Is True when predict() is called after model is trained.
            finalPredictions = [elem[len(elem)-1][0][0] for elem in allPredictions]
            return np.array(finalPredictions)
        
        else: #True If predict() is called during training, 
            return np.array(allPredictions) #and thus all intermediate predictions are necessery to keep for usage in backpropagation.
        

        
# --- Some utility functions 

def binary_accuracy(y_true, y_pred, threshold=0.5):
    """
    Computes binary classification accuracy
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    y_pred_thresholded = (y_pred >= threshold).astype(float)
    correct_predictions = y_pred_thresholded == y_true 
    return correct_predictions.mean()
    

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """
    Computes binary cross entropy 
    
    Args:
        y_true (array<m>): m 0/1 floats with ground truth labels
        y_pred (array<m>): m [0,1] floats with "soft" predictions
        
    Returns:
        Binary cross entropy averaged over the input elements
    """
    assert y_true.shape == y_pred.shape
    y_pred = np.clip(y_pred, eps, 1 - eps)  # Avoid log(0)
    return - np.mean(
        y_true * np.log(y_pred) + 
        (1 - y_true) * (np.log(1 - y_pred))
    )


def sigmoid(x):
    """
    Applies the logistic function element-wise
    
    Hint: highly related to cross-entropy loss 
    
    Args:
        x (float or array): input to the logistic function
            the function is vectorized, so it is acceptible
            to pass an array of any shape.
    
    Returns:
        Element-wise sigmoid activations of the input 
    """
    return 1. / (1. + np.exp(-x))

def relu(x):
    print(x)
    return max(0, x)

        