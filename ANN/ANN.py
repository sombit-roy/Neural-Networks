import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

df = pd.read_csv('heart_dataset.csv')
heart_df = shuffle(df, random_state=2)

# Normalized values only for those columns which do not have categorical data, that is age, resting blood pressure, serum cholestrol, maximum heart rate achieved, and ST depression induced by exercise

col = heart_df.columns
col_list = ["age","trestbps","chol","thalach","oldpeak"]

heart_df_norm = heart_df
for i in col:
    if i in col_list:
        heart_df_norm[i] = (heart_df[i] - np.mean(heart_df[i],axis=0)) / pd.DataFrame.std(heart_df[i],axis=0)

X = np.copy(heart_df_norm.iloc[:,0:13])
Y = np.copy(heart_df_norm.iloc[:,13:14])

# 85:15 ratio of train : test to divide the datasets

X_train = np.array(X[:258])
X_test = np.array(X[258:])
Y_train = Y[:258].reshape(1,-1)
Y_test = Y[258:].reshape(1,-1)

def Initialization(I, H1, H2, O):
    
    tln = []
    W1 = np.random.rand(H1*I)*0.01
    W1 = np.reshape(W1, newshape = (H1,I))
    tln.insert(0,W1)
    b1 = np.zeros((H1,1))
    tln.insert(1,b1)
    W2 = np.random.rand(H2*H1)*0.01
    W2 = np.reshape(W2, newshape = (H2,H1))
    tln.insert(2,W2)
    b2 = np.zeros((H2,1))
    tln.insert(3,b2)
    W3 = np.random.rand(O*H2)*0.01
    W3 = np.reshape(W3, newshape = (O,H2))
    tln.insert(4,W3)
    b3 = np.zeros((O,1))
    tln.insert(5,b3)
    return tln

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return np.maximum(z,0)

def feed_forward(X, parameters):
    
    l = []
    z1 = np.dot(parameters[0],X) + parameters[1]
    l.insert(0,z1)
    a1 = relu(z1)
    l.insert(1,a1)
    z2 = np.dot(parameters[2],a1) + parameters[3]
    l.insert(2,z2)
    a2 = relu(z2)
    l.insert(3,a2)
    z3 = np.dot(parameters[4],a2) + parameters[5]
    l.insert(4,z3)
    a3 = sigmoid(z3)
    l.insert(5,a3)
    return a3,l

eps = 1e-8

def loss_compute(y_pred, yd):
    
    m = len(y_pred[0])
    y1 = np.multiply(-yd,np.log(y_pred+eps))
    y2 = np.multiply(1-yd,np.log(1-y_pred+eps))
    y = np.subtract(y1,y2)
    loss = np.sum(y)
    loss /= m
    return loss

def regularization_L2(lmbda, W1, W2, W3, m):
    
    W1_sq = np.multiply(W1,W1)
    W2_sq = np.multiply(W2,W2)
    W3_sq = np.multiply(W3,W3)
    L2_cost = (np.sum(W1_sq) + np.sum(W2_sq) + np.sum(W3_sq)) * lmbda / m / 2
    return L2_cost

def drelu(z):
    return (np.array(z) >= 0).astype(int)

def dSigmoid(z):
    return np.multiply(sigmoid(z),1-sigmoid(z))

def back_prop_linear(da_layer, z_layer, input, act_fxn, m, lmbda, weight):
    
    if(act_fxn == 'relu'):
        dz = np.multiply(da_layer, drelu(z_layer))
    elif(act_fxn == 'sigmoid'):
        dz = np.multiply(da_layer, dSigmoid(z_layer))
    dW = np.dot(dz,input.T) / m
    dW += lmbda * weight / m
    db = []
    for i in range(0,len(da_layer)):
        temp = []
        temp = np.append(temp,np.sum(dz[i])/m)
        db.insert(i,temp)
    db = np.array(db)
    return dz,dW,db

def back_prop_actf(W_plusone, dz_plusone):
    return np.dot(W_plusone.T,dz_plusone)

def Backpropagation1(X, yd, l, y_pred, parameters, lmbda):
    
    da_3 = -np.divide(yd, l[5]) + np.divide(1-yd, 1-l[5])
    dz_3, dW_3, db_3 = back_prop_linear(da_3, l[4], l[3], 'sigmoid' , len(da_3[0]), lmbda, parameters[4])
    da_2 = back_prop_actf(parameters[4],dz_3)
    dz_2, dW_2, db_2 = back_prop_linear(da_2, l[2], l[1], 'relu' , len(da_2[0]), lmbda, parameters[2])
    da_1 = back_prop_actf(parameters[2],dz_2)
    dz_1, dW_1, db_1 = back_prop_linear(da_1, l[0], X, 'relu' , len(da_1[0]), lmbda, parameters[0])
    grad = {}
    grad['dW1'] = dW_1
    grad['dW2'] = dW_2
    grad['dW3'] = dW_3
    grad['db1'] = db_1
    grad['db2'] = db_2
    grad['db3'] = db_3
    return grad

def training(X, yd, parameters, eta=0.05, num_iters=3000, lmbda=0.1):
    
    losses = np.zeros(num_iters)
    for i in range(0,num_iters):
        y_pred, l = feed_forward(X, parameters)
        grad = Backpropagation1(X, yd, l, y_pred, parameters, lmbda)
        losses[i] = loss_compute(y_pred,yd) + regularization_L2(lmbda, parameters[0], parameters[2], parameters[4], len(X[0]))
        parameters[0] = np.subtract(parameters[0],eta*grad['dW1'])
        parameters[1] = np.subtract(parameters[1],eta*grad['db1'])
        parameters[2] = np.subtract(parameters[2],eta*grad['dW2'])
        parameters[3] = np.subtract(parameters[3],eta*grad['db2'])
        parameters[4] = np.subtract(parameters[4],eta*grad['dW3'])
        parameters[5] = np.subtract(parameters[5],eta*grad['db3'])
    return losses, parameters

np.random.seed(4)
parameters = Initialization(13, 100, 100, 1)
losses, parameters_final = training(X_train.T, Y_train, parameters, eta=0.05, num_iters=3000, lmbda=0.1)

plt.plot(losses)
plt.ylabel('loss')
plt.xlabel('iterations')
plt.show()

def predict(X, yd, parameters_final):
    
    y_forward, l = feed_forward(X, parameters_final)
    n = len(y_forward)
    m = len(y_forward[0])
    output = (y_forward > 0.5).astype(int)
    total = 0
    for i in range(0,n):
        for j in range(0,m):
            if(output[i][j] == yd[i][j]):
                total += 1
    acc = total/m
    return acc

acc = predict(X_test.T, Y_test, parameters_final)
print("Accuracy on test dataset = ", acc)