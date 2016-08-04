__author__ = 'Jernej'

import time
import Orange
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


def load_data():
    """Loads the data. Returns one-column matrix X and vector y."""

    y = np.loadtxt("alc_elim.dat.txt")[:,1:2]
    X = np.loadtxt("alc_elim.dat.txt")[:,0:1]

    return y,X

def add_constant_column(X):
    """Adds a row of 1s to X and returns augmented matrix."""
    X = np.array(X)
    return np.append(X,np.ones((X.shape[0],1)),axis=1)



def gradient(X, y, theta):
    """Return a gradient using an analytically computed function."""
    k = analytical(X,y)
    return k[0]*theta + k[1]


def grad_approx(J,X, y, theta, eps=1e-1):
    """Returns a gradient of function J using finite difference method."""
    return np.array([(J(theta+e,X,y) - J(theta-e,X,y))/(2*eps)
                     for e in np.identity(len(theta)) * eps])


def gradient_descent(X, y, alpha=0.1, epochs=100000):
    """Return parameters of linear regression by gradient descent."""
    theta0 = 0
    theta1 = 0
    for i in range(epochs):
        sum0 = sum((theta0 + theta1*X) - y)[0]
        sum1 = sum(((theta0 + theta1*X) - y)*X)[0]
        theta0 = theta0 - (alpha*sum0)/len(y)
        theta1 = theta1 - (alpha*sum1)/len(y)
    return theta0,theta1


def plot_graph(X, y, thetas, filename="tmp.pdf"):
   plt.xlabel('Breath alcohol elimination rates (mg per litre per hour)')
   plt.ylabel('Blood alcohol elimination rates (g per litre per hour)')
   plt.scatter(X[:,0], y)
   plt.plot(thetas, gradient(X,y,thetas));
   plt.show()



def analytical(X, y):
    """An analytical solution for the linear regression."""
    return np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y))

def J(theta,X,y):
    """Return a value of a optimization function for X, y, and theta."""
    return sum((y - (theta[0] + theta[1]*X))**2)/len(y)



def dJ(theta,X,y):
     return np.array([sum(theta[0] + theta[1]*X - y)/len(y), sum((theta[0] + theta[1]*X - y)*X)/len(y)])[:,0]


y, X = load_data()
X1 = add_constant_column(X)
theta0 = np.array([0,0])



start = time.time()
print "Gradientni spust: "
print gradient_descent(X,y,alpha=0.5,epochs=100000)
end = time.time()
print "Gradientni spust cas: "
print end - start
print "Analiticna resitev: "
start = time.time()
print analytical(X1, y)
end = time.time()
print "Analiticna resitev cas: "
print end - start
print "Optimizacija L-BFGS: "
start = time.time()
print fmin_l_bfgs_b(J,theta0,dJ,args=(X,y))
end = time.time()
print "Optimizacija L-BFGS cas: "
print end - start



thetas = np.arange(0.04, 0.12, 0.00001)
plot_graph(X1,y,thetas)




# load the data
# add the constand column to X
# compare analytical computation of gradient with that by finite differences
# compute theta by gradient descent
# plot the graph h(x) (show points and the regression line)
# compare computed theta with the analytical solution and the one obtained
# by lbfgs

