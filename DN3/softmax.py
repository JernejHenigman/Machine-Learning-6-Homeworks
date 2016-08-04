import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import pickle
import time

np.seterr(all="ignore")

def h0(theta,X,y):

    len = y.size
    index = np.arange(len)
    y = y.astype(int)

    num1 = np.exp(X.dot(theta.T))
    denom1 = 1/np.sum(np.exp((X.dot(theta.T))),1)

    h = (num1 * denom1[:,None])
    r = h[index,y[index]]

    return r

def h1(theta,X,y):

    y = y.astype(int)
    n_class = len(np.unique(y))
    theta = np.reshape(theta,(n_class,X.shape[1]))
    num = np.exp(X.dot(theta.T))
    denom = 1/np.sum(np.exp((X.dot(theta.T))),1)
    r = num * denom[:,None]
    return r

def cost(theta, X, y):
  theta = np.reshape(theta,(len(np.unique(y)),X.shape[1]))
  y = y.astype(int)
  h = h0(theta,X,y)
  r = -sum(np.log(h))

  return r


def approx_grad(theta, X, y):
     """Returns a gradient of function J using finite difference method."""
     grad_a =  np.array([(cost(theta + e, X, y) - cost(theta - e, X, y)) / (2 * 1e-5)
                      for e in np.identity(len(theta)) * 1e-5])
     return grad_a


def grad(theta, X, y):

    n_class = len(np.unique(y))
    theta = np.reshape(theta,(n_class,X.shape[1]))
    y = y.astype(int)
    h = h1(theta,X,y)

    zeros = np.zeros((X.shape[0],n_class))
    index = np.arange(X.shape[0])
    zeros[index,y] = 1
    zeros = zeros - h
    g = -(zeros.T.dot(X).flatten())

    return g

def fit(X, y):
    X = np.insert(X,0,1,axis=1)
    res = fmin_l_bfgs_b(lambda theta, X=X, y=y: cost(theta,X,y),
                        np.zeros((X.shape[1]) * len(np.unique(y))),
                        lambda theta, X=X, y=y: grad(theta,X,y))
    return np.reshape(res[0],(len(np.unique(y)),X.shape[1]))

def predict(theta, X):
    X = np.insert(X,0,1,axis=1)
    return X.dot(theta.T)

def cost_reg(theta, X, y, _lambda):
    theta = np.reshape(theta,(len(np.unique(y)),X.shape[1]))
    y = y.astype(int)
    n_class = len(np.unique(y))
    h = h0(theta,X,y)
    r = -sum(np.log(h)) + np.sum(np.square(theta[:,1:n_class]))*(_lambda/2)
    return r

def grad_reg(theta, X, y, _lambda):
    n_class = len(np.unique(y))
    theta = np.reshape(theta,(n_class,X.shape[1]))
    y = y.astype(int)
    h = h1(theta,X,y)

    zeros = np.zeros((X.shape[0],n_class))
    index = np.arange(X.shape[0])
    zeros[index,y] = 1
    zeros = zeros - h
    g = -((zeros.T.dot(X)+theta*_lambda).flatten())
    return g

def approx_grad_reg(theta, X, y, _lambda):
    """Returns a gradient of function J using finite difference method."""
    grad_a =  np.array([(cost_reg(theta + e, X, y, _lambda) - cost_reg(theta - e, X, y, _lambda)) / (2 * 1e-5)
                      for e in np.identity(len(theta)) * 1e-5])
    return grad_a

def fit_reg(X, y, _lambda):
    X = np.insert(X,0,1,axis=1)
    res = fmin_l_bfgs_b(lambda theta, X=X, y=y: cost_reg(theta,X,y,_lambda),
                        np.zeros((X.shape[1]) * len(np.unique(y))),
                        lambda theta, X=X, y=y: grad_reg(theta,X,y, _lambda))
    return np.reshape(res[0],(len(np.unique(y)),X.shape[1]))

def predict_reg(theta, X):
    X = np.insert(X,0,1,axis=1)
    return X.dot(theta.T)

# full train set - 42k samples
#trainX = pickle.load(open("trainX5.p", "rb"))

# full train class set - 42k samples
#trainY = pickle.load(open("trainY5.p", "rb"))

# full test set - 28k samples
#testX = pickle.load( open( "testX5.p", "rb" ))


