import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("./synthetic_dataset.csv")
color = np.where(data["y"]==1, "red", "blue")
data.insert(0,"x0",[1]*data.shape[0])
plt.scatter(list(data['x1']), list(data['x2']), c=color)

w = [0,1,1]
misclassified_point = dict()
datay = list(data["y"])

def sign(x):
  if x >= 0:
    return 1
  else:
    return -1

def get_sign(x,w):
  dot_product = x[0]*w[0] + x[1]*w[1] + x[2]*w[2]
  return sign(dot_product)

def classify(w):
    global misclassified_point, datay
    classification = []
    misclassified_point = dict()
    for i in range(data.shape[0]):
        classification.append(get_sign(data.iloc[i], w))
        if(datay[i] != classification[i]):
            misclassified_point[i] = classification[i]
            break
    print(misclassified_point)
    

def perceptron(p):
    global w
    x0 = data.iloc[p][0]
    x1 = data.iloc[p][1]
    x2 = data.iloc[p][2]
    #print(list(misclassified_point.keys())[0])
    print(datay[p])
    y = datay[p]

    w[0] = w[0] + x0*y
    w[1] = w[1] + x1*y
    w[2] = w[2] + x2*y

    classify(w)
    print(w)

def plot_it(w):
  # print ("For iteration", i, " and weight", w, " our classification result is: ")
  col = data['y'].map({-1:'b', 1:'r'})
  data.plot.scatter(x='x1', y='x2', c=col)
  axes = plt.gca()
  axes.set_ylim(-10,10)

  x_vals = np.array(axes.get_xlim())
  print(x_vals)
  y_vals = -w[0]/w[2] - w[1]/w[2] * x_vals
  plt.plot(x_vals, y_vals, '--')

classify(w)
while(len(misclassified_point)):
    plot_it(w)
    perceptron(list(misclassified_point.keys())[0])