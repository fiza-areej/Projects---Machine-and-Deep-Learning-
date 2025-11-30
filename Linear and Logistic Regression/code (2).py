#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import seaborn as sns
import pickle
import os

# function for normalizing the dataset
def normalize(X):
    mean = torch.mean(X, dim=0)
    std = torch.std(X, dim=0)
    X_normalized = (X - mean) / std
    return X_normalized
# train path
train = pd.read_csv('C:\\Users\\Usman\\Downloads\\Deep learning assignment\\titanic dataset\\train.csv')

# preprocessing
   
drop_columns = ['Name', 'Ticket', 'Cabin', 'PassengerId']
train.drop(columns=drop_columns, inplace=True)

age = train['Age'].mode()[0]
train.fillna({'Age': age}, inplace=True)  # Filling missing values in Age column

encoding = ['Sex', 'Embarked']
train = pd.get_dummies(train, columns=encoding)  # One-hot encoding categorical variables

# Convert all columns to integer type
encoded_train = train.astype(int)




x = encoded_train[['Pclass','Age','SibSp','Parch','Fare','Sex_female','Sex_male','Embarked_C','Embarked_Q', 'Embarked_S']]  # Features are all columns except the target
y = encoded_train[['Survived']] 
X = torch.tensor(x.values)
Y = torch.tensor(y.values)
trainX = X.float()
trainY = Y.float()
# testpath
test = pd.read_csv('C:\\Users\\Usman\\Downloads\\Deep learning assignment\\titanic dataset\\test.csv')
target = pd.read_csv('C:\\Users\\Usman\\Downloads\\Deep learning assignment\\titanic dataset\\gender_submission.csv')
# print(test)


drop_columns = ['Name', 'Ticket', 'Cabin', 'PassengerId']
test.drop(columns=drop_columns, inplace=True)

# Fill missing values in the 'Age' column with the mode
age_mode = test['Age'].mode()[0]
test['Age'].fillna(age_mode, inplace=True)
fare_mode = test['Fare'].mode()[0]
test['Fare'].fillna(fare_mode, inplace=True)
# print(test.info())
# One-hot encode categorical variables
encoding = ['Sex', 'Embarked']
test = pd.get_dummies(test, columns=encoding)
data = ['Sex_female' , 'Sex_male',  'Embarked_C', 'Embarked_Q',  'Embarked_S']
# print(test)
for col in data:
    test[col] = test[col].astype(int)
# # # Convert all columns to integer type
# test = test.astype(int)
# print(test)

print(test)

columns = [ 'PassengerId']
target.drop(columns=columns, inplace=True)
# converting dataintotensors
X = torch.tensor(test.values)
Y = torch.tensor(target.values)

testX = X.float()
testY = Y.float()
 


normalize(testX)

nan_indices = torch.isnan(testX)
# print(nan_indices[-1:32])
# hyperparameters
normalize(trainX)
input_size = 10
batch_size = 32
n_epochs = 150
lr = 0.001

train_size = int(0.8 * len(trainX))  # 80% for training
val_size = len(trainX) - train_size   # Remaining 20% for validation
print(train_size,val_size)
train_X, val_X = trainX[:train_size], trainX[train_size:]
train_Y, val_Y = trainY[:train_size], trainY[train_size:]
test_X,test_Y = testX,testY
print(test_X.shape)
# Split dataset into training and validation sets
train_data = TensorDataset(train_X, train_Y)
val_data = TensorDataset(val_X, val_Y)
test_data = TensorDataset(test_X, test_Y)
# dataloaders
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data,shuffle=False)


# def logistic_regression(n_features):
#     model = nn.Sequential(
#         nn.Linear(n_features, 1),
#         nn.Sigmoid()
#     )
#     return model
# model = logistic_regression(10)
# logisticregressionclass
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
    
model = LogisticRegressionModel(10)

def main():
    loss_ =  torch.nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    train = []
    val = []
    for epoch in range(n_epochs):
        train_loss = 0.0
        val_loss = 0.0
        for i, data in enumerate(train_loader):
    #         print(data)
            model.train()
            optimizer.zero_grad()
            x,y = data
    #         print(x)
    #         print(y)

        # Make predictions for this batch
            logits = model(x)

        # Compute the loss and its gradients
            loss = loss_(logits, y)
            loss.backward()

        # Adjust learning weights
            optimizer.step()
            value = loss.item()
            train_loss += value
        average_epoch_train_loss = train_loss / len(train_loader)
        train.append(average_epoch_train_loss)
    #     print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {average_epoch_train_loss}")

        for i, data in enumerate(val_loader): 
            model.eval()
            x,y = data
            # Make predictions for this batch
            logits = model(x)
            loss = loss_(logits, y)
            value = loss.item()
            val_loss += value
        average_epoch_val_loss = val_loss / len(val_loader)
        val.append(average_epoch_val_loss)
    #     print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {average_epoch_val_loss}")
        print(f'Train_loss : {average_epoch_train_loss:.4f},Validation_loss : {average_epoch_val_loss:.4f}')

#         Visualization of the training and vaidation loss

    x_axis = [i for i in range(len(train))]
    plt.plot(x_axis, train, label='Train Loss')
    plt.plot(x_axis, val, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

#     Testing function



    # Testing function
    model.eval()
    test_loss = 0.0
    y_hat = []
    y_true = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):

            x,y = data
            output_true = y.squeeze().tolist()
            y_true.extend(output_true)

            logits = model(x)
            preds = torch.round(logits)
            output_hat = preds.squeeze().tolist()
            
            y_hat.extend(output_hat)
            loss = loss_(logits, y)

            value = loss.item()

            test_loss += value
            
    
    
    average_test_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(y_hat, y_true)
    f_1 = f1_score(y_hat, y_true)
    c_m = confusion_matrix(y_hat, y_true)

    print(c_m)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f_1:.4f}')
    train_model = pickle.dump(model, open('model.pkl', 'wb'))
if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




