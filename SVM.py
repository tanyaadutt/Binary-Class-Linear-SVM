# importing libraries
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from cvxopt import matrix as cm
from cvxopt.solvers import qp
from libsvm.svmutil import *

# reading in csv
train_data = pd.read_csv("Data/train.csv", header=None)
test_data = pd.read_csv("Data/test.csv", header=None)


# Primal Form
def solve_primal(X, y, C):
    y1 = y.reshape(-1, 1)
    y1[y1 == 0] = -1  # changing 0 to -1 since our training data has labels 0 and 1
    m, n = X.shape  # storing array dimension

    # To solve primal form using cvxopt, we need 4 matrix parameters P, q, G and h
    # calculating P
    P = np.zeros((n + m + 1, n + m + 1))
    for i in range(n):
        P[i, i] = 1
    P = cm(P)

    # calculating q
    q = np.zeros((m + n + 1, 1))
    for i in range(n + 1, n + m + 1):
        q[i, 0] = C * 1
    q = cm(q)

    # calculating G
    G = np.zeros((2 * m, n + m + 1))
    G[:m, 0:n] = y1 * X
    G[:m, n] = y1.T
    G[:m, n + 1:] = np.identity(m)
    G[m:, n + 1:] = np.identity(m)
    G = cm(G * -1)

    # calculating h
    h = np.zeros((2 * m, 1))
    h[:m] = -1
    h = cm(h)

    result = qp(P, q, G, h)  # solving quadratic programming problem to minimize w, b and slack variable E
    y1[y1 == -1] = 0  # changing back -1 to 0

    return result


# getting primal form w and b
def get_primal_parameters(X, a):
    n = X.shape[1]  # storing number of features
    w = np.array(a[:n])  # weight
    b = a[n]  # intercept
    return w, b


# Dual Form
def solve_dual(X, y, C):
    yd = y
    yd[yd == 0] = -1  # changing 0 to -1 since our training data has labels 0 and 1
    m = X.shape[0]  # storing number of rows

    # To solve dual form using cvxopt, we need all of the 6 matrix parameters P, q, G, h, A and b
    # calculating P
    xx = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            xx[i, j] = np.dot(X[i], X[j])

    yy = np.outer(yd, yd)

    P = cm(yy * xx)

    # calculating q
    q = cm(np.ones((m, 1)) * -1)

    # calculating G
    i1 = np.identity((m)) * -1
    i2 = np.identity((m))
    G = cm(np.concatenate((i1, i2), axis=0))

    # calculating h
    z = np.zeros((m, 1))
    c = (np.ones((m, 1)) * C)
    h = cm(np.concatenate((z, c), axis=0))

    # calculating A
    A = cm(yd.reshape(1, -1))

    # calculating b
    b = cm(np.zeros(1))

    result = qp(P, q, G, h, A, b)  # solving quadratic programming problem to minimize alpha
    yd[yd == -1] = 0  # changing back -1 to 0
    return result


# getting dual form w and b
def get_dual_parameters(X, y, a):
    X_temp = pd.DataFrame(X)
    y1 = y.reshape(-1, 1)
    y1[y1 == 0] = -1  # changing 0 to -1
    S = (a > 1e-8).reshape(-1, )  # support vectors; threshold = 1e-8
    w = np.dot(X.T, a * y1)  # weight
    b = y1[S] - np.dot(X[S], w)  # intercepts
    b = np.mean(b)  # taking mean of intercepts as final intercept
    y1[y1 == -1] = 0  # changing back -1 to 0

    # getting idx of support vectors
    dualSVIdx = np.asarray(np.where(S == True))
    idx = []
    for i in range(dualSVIdx.shape[1]):
        idx.append(dualSVIdx[0][i])  # storing support vector index in a list
    SV = X_temp.iloc[idx][:].reset_index(drop=True)  # support vectors
    return w, b, SV


# Predict Results
def predict(X_test, w, b):
    prediction = b + np.dot(X_test, w)  # yi = b + xi.w
    prediction[prediction > 0] = 1  # classifying values above 0 as of class 1
    prediction[prediction < 0] = -1  # classifying values below 0 as of class -1
    return prediction


# Model accuracy
def get_accuracy(y_test, y_predicted):
    y_predicted[y_predicted == -1] = 0  # changing value -1 to 0 as our training data has label 0 and 1 only
    result = pd.DataFrame()
    result['Expected'] = y_test
    result['Predicted'] = y_predicted
    accuracy = ((len(result) - len(result[result['Expected'] != result['Predicted']])) / len(result)) * 100
    return accuracy


# # Solutions
# storing training X and y
X = train_data.drop([0], axis=1).to_numpy()
y = train_data[0].to_numpy()

# storing testing X_test a nd y_test
X_test = test_data.drop([0], axis=1).to_numpy()
y_test = test_data[0].to_numpy()

# primal solution
primal_solution = solve_primal(X, y, C=0.001)
a = np.array(primal_solution['x'])  # minimized w, b and slack variable E
w_primal, b_primal = get_primal_parameters(X, a)  # getting w and b
y_predicted_primal = predict(X_test, w_primal, b_primal)  # predicted result for X_test
model_accuracy_primal = get_accuracy(y_test, y_predicted_primal)  # primal form accuracy

# dual solution
dual_solution = solve_dual(X, y, C=0.001)
alpha = np.array(dual_solution['x'])  # minimized alpha
w_dual, b_dual, SV_dual = get_dual_parameters(X, y, alpha)  # getting w, b and support vectors SV
y_predicted_dual = predict(X_test, w_dual, b_dual)  # predicted result for X_test
model_accuracy_dual = get_accuracy(y_test, y_predicted_dual)  # dual form accuracy

# libsvm Solution
# converting training and testing data set to lists as libsvm trains data in lists only
X_libsvm = X.tolist()
y_libsvm = y.tolist()
X_libsvm_test = X_test.tolist()
y_libsvm_test = y_test.tolist()

parameters = svm_parameter("-q")
problem = svm_problem(y_libsvm, X_libsvm)
parameters.C = 0.001
parameters.gamma = 0  # 0 as linear svm has no gamma
parameters.kernel_type = 0  # 0 represents linear svm

model = svm_train(problem, parameters)
y_predicted_libsvm, p_acc, p_val = svm_predict(y_libsvm_test, X_libsvm_test, model)

# Parameters libsvm
# weight
w_libsvm = -np.matmul(np.array(X_libsvm)[np.array(model.get_sv_indices()) - 1].T, model.get_sv_coef())

# intercept
b_libsvm = model.rho.contents.value

# support vectors
SV_libsvm = pd.DataFrame(model.get_SV()).drop([-1], axis=1)  # dropping column -1

# concatenating predictions
final_result = pd.DataFrame()
final_result['Expected'] = y_test
final_result['Primal'] = y_predicted_primal
final_result['Dual'] = y_predicted_dual
final_result['libsvm'] = y_predicted_libsvm
print("Final Prediction Result:\n", final_result)

# concatenating w
w = pd.DataFrame()
w['Primal'] = pd.DataFrame(w_primal)[0]
w['Dual'] = pd.DataFrame(w_dual)[0]
w['libsvm'] = pd.DataFrame(w_libsvm)[0]
print("weights w: \n", w)

# printing b
print("primal intercept b: ", b_primal[0])
print("dual intercept b: ", b_dual)
print("libsvm intercept b: ", b_libsvm)

# printing accuracies
print("\nprimal accuracy: ", model_accuracy_primal)
print("dual accuracy: ", model_accuracy_dual)
print("libsvm accuracy: ", p_acc[0])

# getting index of all the misclassified data sets
Idx_primal = final_result[final_result['Expected'] != final_result['Primal']].index.tolist()  # misclassified by primal
Idx_dual = final_result[final_result['Expected'] != final_result['Dual']].index.tolist()  # misclassified by dual
Idx_libsvm = final_result[final_result['Expected'] != final_result['libsvm']].index.tolist()  # misclassified by libsvm

# checking if all 3 models misclassified same data points
print("All model misclassified same data points: ", Idx_primal == Idx_dual == Idx_primal)
print("Number of misclassifications:", len(Idx_primal))

IDX = pd.DataFrame()      # missclassified datapoints index
IDX['p'],IDX['d'],IDX['l'] = Idx_primal, Idx_dual, Idx_libsvm
print("Misclassified data points Index:-\n",IDX)  # dual classified 1193 index datapoint correctly but misclassified datapoint with index 408

# checking if w of primal and dual same
print("w_primal and w_dual same: ", (w_primal - w_dual).all() == 0)

# checking if b of primal and dual same
print("b_primal and b_dual same: ", (b_primal - b_dual)[0] == 0)
if (b_primal - b_dual)[0] != 0:
    if (b_primal - b_dual)[0] > 0:
        print("Difference between b_primal and b_dual ", (b_primal - b_dual)[0])
    else:
        print("Difference between b_primal and b_dual ", -((b_primal - b_dual)[0]))

# checking if w of dual and libsvm same
print("w_dual and w_libsvm same: ", (w_dual - w_libsvm).all() == 0)

# checking if b of dual and libsvm same
print("b_dual and b_libsvm same: ", b_dual - b_libsvm == 0)
if b_dual - b_libsvm != 0:
    if b_dual - b_libsvm > 0:
        print("Difference between b_dual and b_libsvm", b_dual - b_libsvm)
    else:
        print("Difference between b_dual and b_libsvm", -(b_dual - b_libsvm))

# support vectors
print("libsvm Support Vectors:\n")
# decreasing column names by 1 to make it same as dual form
for i in SV_libsvm:
    SV_libsvm.rename(columns={i: i - 1}, inplace=True)
print(SV_libsvm)
print("Dual form Support Vectors:\n", SV_dual)

# common support vectors
print("There are", pd.merge(SV_libsvm, SV_dual, how='inner', on=[0, 1, 2, 3, 4, 5]).shape[0],
      "common Supports vectors in dual form and libsvm and they are:\n",
      pd.merge(SV_libsvm, SV_dual, how='inner', on=[0, 1, 2, 3, 4, 5]))

# mean squared error for testing
print("Testing error(primal):", mean_squared_error(final_result['Expected'], final_result['Primal']),
      "\nTesting error(dual):", mean_squared_error(final_result['Expected'], final_result['Dual']),
      "\nTesting error(libsvm):", p_acc[1])

#  gap can be calculated by s^T.Z
#  where s and z can be obtained from minimized/maximized solutions of primal and dual form using cvxopt
#  or we can directly get gap by using key 'gap' in minimized solution dictionary returned by solvers.qp
print("Primal form duality gap: ", primal_solution['gap'])
print("Dual form duality gap: ", dual_solution['gap'])
print("Difference between gap of primal and dual form:", primal_solution['gap'] - dual_solution['gap'])
