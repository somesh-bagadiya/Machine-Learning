# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:02:02 2023

@author: somes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./synthetic_dataset_training_100.csv")
testing_data = pd.read_csv("./synthetic_dataset_testing_10000.csv")
data.insert(0,"x0",[1]*data.shape[0]) 
testing_data.insert(0,"x0",[1]*testing_data.shape[0]) 

def updateW(w, x, y):
    x = np.array(x)
    s = np.dot(w,x)
    w = w + learning_rate*(y-s)*x
    return w

def check_miss_points(w):
    for i in range(data.shape[0]):
        x = list(data.iloc()[i])[:3]

        value = np.dot(w, x)
       
        if value * data["y"][i] <= 1:
            return list(data.iloc()[i])
    return [1,1,1,-1]

def plot_graph(w):
    
    fw0 = 0.5
    fw1 = 0.3
    fw2 = 0.4
    fyAxis = []
    
    
    x_vals = list(range(-11,11,1))
    y_vals = []
    for i in x_vals:
        y = -w[0]/w[2] - (w[1]/w[2] * i)
        y_vals.append(y)
        fyAxis.append(-fw0/fw2 - (fw1/fw2*i))
    plt.plot(x_vals, fyAxis, '-')

    plt.axvline(x=0, c="black", label="x=0")
    plt.axhline(y=0, c="black", label="y=0")
    plt.plot(x_vals, y_vals)
    plt.ylim(-11, 11)
    plt.xlim(-11, 11)

    color = np.where(data["y"]==1, "red", "blue")
    plt.scatter(list(data['x1']), list(data['x2']), c=color)
    plt.show()

def perceptron_algo(w):
    n = 0
    while n < 1000:
        x = check_miss_points(w)
        y = x[3]
        x = x[:3]
        w = updateW(w,x,y)
        n+=1
    plot_graph(w)
    print("Final Weights (Perceptron):",w)
    return w


def is_misclassified(w, point):
    x = point[0:3]
    y = point[3]
    classified_value = np.dot(w, x)
    if classified_value > 0:
        classified_value = 1
    else:
        classified_value = -1
    
    return classified_value != y

def count_misclass(final_weights):
    count_of_misclassified = 0
    for i in range(len(testing_data)):
        misclassified = is_misclassified(final_weights, testing_data.iloc[i].values)
        if misclassified==True:
            count_of_misclassified+=1  
    print("percent of accuracy:",(1 - count_of_misclassified/len(testing_data))*100, "%")

learning_rate = 100
w = [0,0,0]
w = perceptron_algo(w)
count_misclass(w)

learning_rate = 1
w = [0,0,0]
w = perceptron_algo(w)
count_misclass(w)

learning_rate = 0.01
w = [0,0,0]
w = perceptron_algo(w)
count_misclass(w)

learning_rate = 0.0001
w = [0,0,0]
w = perceptron_algo(w)
count_misclass(w)

learning_rate = 0.052
w = [0,0,0]
w = perceptron_algo(w)
count_misclass(w)










