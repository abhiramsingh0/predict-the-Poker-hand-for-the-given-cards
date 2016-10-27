import numpy as np
import matplotlib.pyplot as plt
import csv

print "reading data from files"
# read input files in X and y
my_data = np.genfromtxt('train.csv', delimiter=',')
my_data_test = np.genfromtxt('test1.csv', delimiter=',')
print "file read success"

X = my_data[1:len(my_data),0:-1]
train_size =  len(X)
y_ = my_data[1:len(my_data),-1]
X_test = my_data_test[1:len(my_data_test),1:len(my_data_test[1,:])]

# randomly permute train data before split
perm = np.arange(train_size)
np.random.shuffle(perm)
X = X[perm]
y_ = y_[perm]

def feature_noramlize(X):
    mean = X.mean(0)
    std = np.std(X,0)
    X = (X - mean) / (std)
    return X

print "normalizing input features"
# applying z-score normalization
X = feature_noramlize(X)
X_test = feature_noramlize(X_test)

print "converting target values to one hot vector"
# convert target values to one hot vector
y_ = y_.astype(int)
y = np.eye(np.max(y_) + 1)[y_]


# neural network design
hidden_layers = 1
nurns_hidden_lyr = 200
nurns_input_lyr = len(X[0,:])
nurns_output_lyr = len(np.unique(y_))
print ("hidden neurons: %d"%nurns_hidden_lyr)

# randomly initialize weights including bias term
def initial_w(wout, win):
    init = 0.001
    weights = np.random.rand(wout, win) * 2 * init - init
    return weights

print "randomly initialize weights"
theta1 = initial_w(nurns_hidden_lyr, (nurns_input_lyr+1))
theta2 = initial_w(nurns_output_lyr, (nurns_hidden_lyr+1))

# insert bias terms
temp = np.ones((len(X),1))
X = np.concatenate((temp, X),1)
temp = np.ones((len(X_test),1))
X_test = np.concatenate((temp, X_test),1)

# divide into train, validation and test set
train_size =  int(len(X) * 0.8)
val_size = int(0.7 * (len(X) - train_size))

Xtrain = X[:train_size]
ytrain = y[:train_size]
# val set is used mainly for identifying number of iterations
# and training and network parameters.
Xval = X[train_size:train_size+val_size]
yval = y[train_size:train_size+val_size]

Xtest = X[train_size+val_size:-1]
ytest = y[train_size+val_size:-1]


print "setting network parameters"
# set parameters
reg_para = 0.0
step_size = 0.07
iterations = 500
print ("step_size: %f"%step_size)
print ("iterations: %d"%iterations)

# intialize empty list for error values
# used to identify number of iterations
Jtrain = []
Jval = []
# sigmoid activation function
# works for both scalar and array
def sigmoid(x):
    return (1.0 / (1.0 + np.exp(-x)))

# validation error
def val_error(x, y, t1, t2):
    m = len(x)
    sum2 = np.dot(x, t1.T)
    act2 = sigmoid(sum2)
    act2 = np.concatenate((np.ones((m,1)), act2), 1)
    sum3 = np.dot(act2, t2.T)
    act3 = sigmoid(sum3)
    val_error =  (1.0/2) * np.sum((y - act3)**2)
    return val_error

print "training started"
for j in range(iterations):
    # in each iteration, update for each training example
    for i in range(train_size):
        # extract one input from data set
        x = Xtrain[i]
        t = ytrain[i]

        # --------forward propagation--------
        # calculate linear output at 1st hidden layer
        sum2 = np.dot(theta1, x)
        # sigmoid activation output at 1st hidden layer
        act2 = sigmoid(sum2)
        # add bias unit
        act2_ = np.concatenate(([1], act2))
        # calculate linear sum at output layer
        sum3 = np.dot(theta2, act2_)
        # sigmoid activation at output layer
        act3 = sigmoid(sum3)

        #------ find regularized cost-------
        if (i%200000 == 0):
            error =  (1.0/2) * np.sum((t - act3)**2)
            reg_cost = (reg_para/2.0) * (np.sum(theta1**2)\
                    + np.sum(theta2**2) \
                    - np.sum(theta1[:,0]**2) \
                    - np.sum(theta2[:,0]**2))
            Jtrain.append(error+reg_cost)
            # compute validation error on some examples
            Jval.append(val_error(Xval[0:100],\
                    yval[0:100],theta1,theta2))

        # ----------error backward propagation------------
        # error at output layer
        grad_3 = act3 * (1.0 - act3) * (t - act3)
        # error at hidden layer
        grad_back = np.dot(theta2.T, grad_3)
        grad_2 = act2_ * (1.0 - act2_) * grad_back
        # gradient at output layer
        # commmented part is for regularization
        delta_w2 = np.outer(grad_3, act2_) #+ (reg_para * 
#                np.concatenate((np.zeros((len(theta2),1)),
#                theta2[:,1:len(theta2[0,:])]), 1))
        # gradient at hidden layer
        # commented part is for regularization
        delta_w1 = np.outer(grad_2[1:len(grad_2)], x) #+ 
#                (reg_para * np.concatenate( 
#                    (np.zeros((len(theta1),1)),
#                theta1[:,1:len(theta1[0,:])]), 1))

        # stochastic gradient descent---weight update
        theta2 += (step_size * delta_w2)
        theta1 += (step_size * delta_w1)
    # shuffle X and y randomly before next iteration
        #if (i == (train_size-1)):
    perm = np.arange(train_size)
    np.random.shuffle(perm)
    Xtrain = Xtrain[perm]
    ytrain = ytrain[perm]
    print ("itertion %d completed"%(j+1))

print "training done"

# predict on current input values
def predict(theta1, theta2, X):
    p1 = sigmoid(np.dot(theta1,X))
    temp2 = np.concatenate(([1], p1))
    p2 = sigmoid(np.dot(theta2,temp2))
    return (np.argmax(p2))
#======training set accuract=============
print "predicting values for training set"
pred_class = []
for i in range(train_size):
    pred_class.append(predict(theta1, theta2,\
            Xtrain[i]))
print "prediction done"
accuracy = (pred_class==y_[:train_size])
print ('training set accuracy %f'%(np.mean(accuracy)))

# =======validation set accuracy=============
print "predicting values for validation set"
pred_class1 = []
for i in range(val_size):
    pred_class1.append(predict(theta1, theta2,\
            Xval[i]))
print "prediction done"
accuracy = (pred_class1==y_[train_size:train_size+val_size])
print ('validation set accuracy %f'%(np.mean(accuracy)))

# ===========test set accuracy===============
print "predicting values for test set"
pred_class2 = []
for i in range(len(Xtest)):
    pred_class2.append(predict(theta1, theta2,\
            Xtest[i]))
print "prediction done"
accuracy = (pred_class2==y_[train_size+val_size:-1])
print ('test set accuracy %f'%(np.mean(accuracy)))

#print (y_,pred_class)
plt.plot(range(len(Jtrain)), Jtrain, label="train error")
plt.plot(range(len(Jval)), Jval, label="validation error")
plt.ylabel('trianing and validation squared error')
plt.legend()
plt.show()

#print (y_[:train_size],pred_class)
#========== generating output for test file==========
print "predicting values for 2nd test set"
pred_class3 = []
for i in range(len(X_test)):
    out =  predict(theta1, theta2, X_test[i])
    pred_class3.append(out)
print "prediction done"

#==========write predicted output in file===============
print ('opening file for write')
label = ["id,CLASS"]
with open('output.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=' ')
    spamwriter.writerow(label)
    for i in xrange(0,len(pred_class3)):
        values = [str(i)+','+str(pred_class3[i])]
        spamwriter.writerow(values)
print ('file write complete')
#=======================================================
print theta1[1,0]

