#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import all necessary libraries 
import numpy as np                                    # For matrices and MATLAB like functions                  
from sklearn.model_selection import train_test_split  # To split data into train and test set
# for plotting graphs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_california_housing
import pickle
import os

Viz_Data = True

# mean computation
def set_mean(X):
    mean_values=[]
    shape=X.shape
    for i in range(shape[1]):
        mean_values.append(np.mean(X[:,i]))
        
    return mean_values

#  standarddeviation computation  

def set_standardDeviation(X):
    std_values=[]
    shape=X.shape
    for i in range(shape[1]):
        std_values.append(np.std(X[:,i]))
    
    return std_values


def normalize(X,mean_values,std_values):
    shape=X.shape
    for col in range(shape[1]):
        for row in range(shape[0]):
            X[row,col]=(X[row,col]-mean_values[col])/std_values[col]
            
    return X

def plot_predictions(testY, y_pred):
 
    plt.figure(figsize=(15,8))
    plt.plot(testY.squeeze(), linewidth=2 , label="True")
    plt.plot(y_pred.squeeze(), linestyle="--",  label="Predicted")
    plt.title('Test case')
    plt.legend()
    plt.show()
    

    
def plot_losses(epoch_loss):

    x_axis=[i for i in range(len(epoch_loss))]
    plt.plot(x_axis,epoch_loss)
#     plt.title('Epochloss')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.show()



def data_split(X, Y):
    shape=X.shape
    train=int(shape[0]*0.7)
    val=int(shape[0]*0.2)
    test=int(shape[0]*0.1)
    trainX=X[0:train,:]
    trainY=Y[0:train]
    valX=X[train:(train+val),:]
    valY=Y[train:(train+val)]
    testX=X[(train+val):,:]
    testY=Y[(train+val):]
    
    return trainX,trainY,valX,valY,testX,testY


# # ---
# # 
# # ## Linear Regression using stochastic gradient descent via Numpy Arrays
# # 
##

class linear_regression_network(object):        
   
    
    # Initialize attributes of object
    def __init__(self,  n_features = 8):        
        
        # No. of input features
        self.n_features  = n_features
        # Learnable weights
        self.number_of_weights=8
        self.theta = np.array([np.random.rand() for i in range(self.number_of_weights)])
      

        
    # This function just prints a few properties of object created from this class    
    def __str__(self):
        
        msg = "Linear Regression:\n\nSize of Input = " + str(self.n_features)
                
        return  msg
        
           
    # Read section#5.4.1 for the help   
    def feed_forward(self, train_X):
        # Write linear function here
        self.theta=self.theta.reshape((self.theta.shape[0],1))
        y_hat = np.dot(train_X,self.theta)
        y_hat=y_hat.reshape((y_hat.shape[0],1))

        return y_hat 

    
    
    # Read section#5.4.2 for the help
    def l2_loss(self, train_Y, y_hat):
        y=train_Y
        loss= np.dot(1/(2*len(train_Y)),np.sum(np.square(y_hat-train_Y)))
        return loss
     
    # Read section#5.4.3 for the help
    def compute_gradient(self , train_X , train_Y , y_hat):
        grad=np.dot(train_X.transpose(),(y_hat-train_Y))
        grad=grad.reshape((grad.shape[0],1))
        
        return grad


    def optimization(self , lr, grad):
        self.theta-=lr*grad

        

def test_function(train_model, test_X, test_Y):
    test_pred=train_model.feed_forward(test_X)
    test_loss=train_model.l2_loss(test_pred,test_Y)
    print('Test loss',test_loss)
    return test_pred
    
    
    
    
# Write your code here

def train(model, trainX, trainY, valX, valY, n_epochs, lr):
    # Write your training loop here
    length=len(trainX)
    epoch_loss=[]
    val_loss=[]
    batch_size=16
    num_of_batches=length//batch_size
    for i in range(n_epochs):
        j=0
        train_loss=0
        while j<length-batch_size:
            y_hat=model.feed_forward(trainX[j:j+batch_size,:])
            grad=model.compute_gradient(trainX[j:j+batch_size,:],trainY[j:j+batch_size],y_hat)
            model.optimization(lr,grad)
            train_loss+=model.l2_loss(trainY[j:j+batch_size],y_hat)

            j+=batch_size
            
        epoch_loss.append(train_loss/num_of_batches)
        
        pred_val=model.feed_forward(valX)
        validation_loss=model.l2_loss(valY,pred_val)
        val_loss.append(validation_loss)
        
    return model, epoch_loss, val_loss

def main():
    # Data loader 
    dataset =  fetch_california_housing()
    X = dataset.data
    Y = dataset.target[:,np.newaxis]

    # Split the dataset into training and testing and validation
    train_X, train_Y, val_X, val_Y, test_X, test_Y = data_split(X, Y)
    Me = set_mean(train_X)

    St = set_standardDeviation(train_X)

    # Normalize
    train_X = normalize(train_X, Me, St)
    val_X = normalize(val_X, Me, St)
    test_X =  normalize(test_X, Me, St)

    net = linear_regression_network( n_features = 8 )

    # Train Network using stochastic gradient descent
    lr = 0.00001
    n_epochs=150
    model, epoch_loss, val_loss = train(net, train_X, train_Y, val_X, val_Y, n_epochs, lr)
    print('Epoch loss')
    plot_losses(epoch_loss)
    print('Validation loss')
    plot_losses(val_loss)

    # Save the model
    train_model = pickle.dump(model, open('model.pkl', 'wb'))

#    Load the model
    path=os.path.join(os.getcwd(),'model.pkl')
    train_model = pickle.load(open(path, 'rb'))

    # ## Test prediction Prediction

    y_pred = test_function(train_model, test_X, test_Y)
    plot_predictions(test_Y, y_pred)







if __name__ == "__main__":
    main()



# In[ ]:




