import numpy as np
from sklearn.cross_validation import KFold
from scipy.optimize import fmin_l_bfgs_b
import pickle
import time
from sklearn import preprocessing

np.seterr(all="ignore")


def load_data(train, test):
    # We load data into numpy arrays. We add columns of ones to test and train sets. We get Y vector from train set.
    # We preprocess data (remove unimportant features, normalize data)
    # We craete 3 pickle objects. (TrainX, trainY, testX)
    # This method was used only once. In the future we always use pickle objects to retrive data.

    trainX = np.genfromtxt(train, dtype=int, delimiter=",")
    testX = np.genfromtxt(test, dtype=int, delimiter=",")
    testY = np.array(testX[:, 0])
    trainY = np.array(trainX[:, 0])

    trainX_scaled, testX_scaled = preprocess_data(trainX, testX)
    trainX_scaled[:, 0] = np.ones(len(trainX_scaled))
    testX_scaled[:,0] = np.ones(len(testX_scaled))


    pickle.dump(trainX_scaled, open("trainX.p", "wb"))
    pickle.dump(trainY, open("trainY.p", "wb"))
    pickle.dump(testX_scaled, open("testX.p", "wb"))


def preprocess_data(trainX, testX):
    # We remove unimportant features from test and train set and we also normalize data.
    pixel_sum = trainX.sum(axis=0)
    indexes_to_be_removed = [i for i, v in enumerate(pixel_sum) if v / len(pixel_sum) < 5]
    trainX_f_sel = np.delete(trainX, indexes_to_be_removed, axis=1)
    testX_f_sel = np.delete(testX, indexes_to_be_removed, axis=1)

    trainX_scaled = preprocessing.normalize(trainX_f_sel.astype(float))
    testX_scaled = preprocessing.normalize(testX_f_sel.astype(float))

    return trainX_scaled, testX_scaled


def one_versus_all(trainX, trainY, testX, testY, alpha):
    # With this function we process data as multiclass problem. In each iteration (10) each class digit is set to 0 and rest is set to 1.
    # As a result, accuracy score is returned. Used for cros validation.
    n_classes = 10
    thetas = np.zeros((n_classes, np.shape(trainX)[1]))

    # Learning
    start = time.time()
    for i in range(n_classes):
        tempY = [1 if float(j) == float(i) else 0 for j in trainY]
        thetas[i] = lr(trainX, np.array(tempY), alpha)

    end = time.time()
    print("Learning Time: " + str(end - start))

    y_p = np.zeros((n_classes, len(testX)))

    # Predicting
    for i in range(n_classes):
        y_p[i] = testX.dot(thetas[i])

    # We check how many predictions are actually correct. For score measurment metric accuracy is used.
    y_final_predictions = np.argmax(y_p, axis=0)
    y_final_predictions = np.equal(testY, y_final_predictions).astype(int)
    score = float(sum(y_final_predictions)) / float(len(y_final_predictions))

    return score


def one_versus_all_kaggle(trainX, trainY, testX):
    # With this function we process data as multiclass problem. In each iteration (10) each class digit is set to 0 and rest is set to 1.
    # Function to output predictions to file. Function does not meassure accuracy.
    n_classes = 10
    thetas = np.zeros((n_classes, np.shape(trainX)[1]))

    # Learning
    start = time.time()
    for i in range(n_classes):
        tempY = [1 if float(j) == float(i) else 0 for j in trainY]
        thetas[i] = lr(trainX, np.array(tempY),0.1)
    end = time.time()
    print("Learning Time: " + str(end - start))

    y_p = np.zeros((n_classes, len(testX)))

    # Predicting
    for i in range(n_classes):
        y_p[i] = testX.dot(thetas[i])

    # Output predictions to a file
    y_final_predictions = np.argmax(y_p, axis=0)
    output = open("predictionsLR1", "wb")
    indexes = np.arange(1, len(y_final_predictions) + 1)
    np.savetxt(output, np.c_[indexes, y_final_predictions], fmt='%i', delimiter=',')
    output.close()


def validation(trainX, trainY, n, alpha):
    # We make n-fold cross validation. We return average score of predictions in each validation!
    i = 0
    kf = KFold(len(trainY), n_folds=n)
    scores = np.zeros(n)
    for train, test in kf:
        scores[i] = one_versus_all(trainX[train], trainY[train], trainX[test], trainY[test], alpha)
        print(scores[i])
        i = i + 1

    final_score = np.average(scores)
    return final_score


def select_best_alfa(trainX, trainY, n, alphas):
    # With this function we gather results with cross validation. We select diffent values for regularazation paramater
    # and observe how that affects accuracy result. We return vector of accuracy scores.
    scores = np.zeros(len(alphas))
    for i, alpha in enumerate(alphas):
        scores[i] = validation(trainX, trainY, n, alpha)
        print("iteracija: %i alpha: %i" % (i, alpha))

    return scores


def g(z):
    return 1. / (1 + np.exp(-z))


def J_lr(X, y, theta, alpha):

    yh = g(X.dot(theta))
    r =  -sum(y * np.log(yh) + (1 - y) * np.log(1 - yh)) + alpha * sum(theta ** 2)
    return r


def grad_approx(J, X, y, theta, alpha, eps=1e-1):
    """Returns a gradient of function J using finite difference method."""
    return np.array([(J(X, y, theta + e, alpha) - J(X, y, theta - e,alpha)) / (2 * eps)
        for e in np.identity(len(theta)) * eps])


def dJ_lr(X, y, theta, alpha):
    return ((g(X.dot(theta)) - y).dot(X)) + 2 * alpha * theta


def lr(X, y, alpha):
    res = fmin_l_bfgs_b(lambda theta, X=X, y=y: J_lr(X, y, theta, alpha),
                        np.zeros(X.shape[1]),
                        lambda theta, X=X, y=y: dJ_lr(X, y, theta, alpha))
    return res[0]

def check_gradient(trainX,trainY):
    # With this function we check if the gradient calculated with finite differnce method outputs the same result as derivative of function J_lr.
    # We see that gradients are the same.
    tempY = np.array([1 if float(j) == float(5) else 0 for j in trainY])
    theta = np.zeros((9))

    grad1 = grad_approx(J_lr,trainX[0:1000,1:10],tempY[0:1000],theta,0.01,eps=1e-1)
    grad2 = dJ_lr(trainX[0:1000,1:10],tempY[0:1000],theta,0.01)
    print(grad1)
    print(grad2)


#########################################################################
#                                                                        #
#                              INPUT                                     #
#                                                                        #
#########################################################################




#load_data("train.csv","test.csv")

# full train set - 42k samples
trainX = pickle.load(open("trainX5.p", "rb"))

# full train class set - 42k samples
trainY = pickle.load(open("trainY5.p", "rb"))

# full test set - 28k samples
testX = pickle.load( open( "testX5.p", "rb" ) )

#check_gradient(trainX,trainY)

# regularization parameters
alphas = np.arange(0.01, 0.20, 0.01)


np.set_printoptions(precision=3)
n_samples = 1000
n_fold_cross_vali = 3
scores = select_best_alfa(trainX[0:n_samples], trainY[0:n_samples], n_fold_cross_vali, alphas)
print(list(zip(alphas, scores)))
