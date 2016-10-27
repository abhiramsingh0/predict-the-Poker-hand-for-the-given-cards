# predict-the-Poker-hand-for-the-given-cards
Each record is an example of a hand consisting of five playing cards drawn from a standard deck of 52. Each card is described using two attributes (suit and rank), for a total of 10 predictive attributes

This task was performed to complete the assignment given in
IITB-CSE, CS725 course, autumn 2016.

this file gives description of 2 layer neural network having 
one hidden layer. Neural network is implemented from scratch.

This code is implementation of neural network algorithm given
in book Machine Learning by Tom Mitchell.

This file gives description of code done for Assignment 2

main.py contains all necessary code.

-------------------------------------------------------
Command to run the program
---------------------------------------------------------
from linux/unix shell:
python main.py

-----------------------------------------------------
Description of program
----------------------------------------------------
Neural Network Implementation

* lines 6 to 14
    fetch data from corresponding files
* lines 17-31
    randomly shuffle data and feature normalize
* lines 35-36
    convert target values to one hot vector
* lines 40-43
    neural network design
* lines 47-54
    random initialization of weights
* lines 57-60
    insert bias term in the data points
* lines 63-74
    divide data into train, validation and test set
* lines 79-81
    initialization of training parameter
* lines 106-167
    -neural network training using back propagation algorithm as...
        described in the book MACHINE LEARNING by Tom Mitchell.
    -stochastic gradient descent is used for weight update i.e
        for each training example weight is updated.
    -lines 113-123
        forward propagation using sigmoid function as activation.
    -lines 126-135
        training and validation error is calculated.
    -lines 139-150
        error is backpropagated and gradient is computed at each layer.
    -lines 156,157
        weight update using gradient descent.
    -lines 160-163
        random shuffle of weights
* lines 172-end
    predicting values for training,validation and test sets
