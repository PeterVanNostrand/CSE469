import numpy as np
import matplotlib.pyplot as plt
import sys as sys
import csv as csv

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def frange(x, y, jump):
    while x < y:
        yield x
        x += jump

def calc_cost(p, y):
    '''
    computes the cost of the predicted labels relative to the acutal labels
    ## Parameters
        - p: numpy array of predicted likelihood for each sample to be positive
        - y: numpy array of the actual labels for these samples
    ## Returns
        - cost: the loss for every sample in the training set
    '''
    a = np.multiply(y, np.log(p)) # y*log(p)
    b = np.multiply((1 - y), np.log(1 - p)) # (1-y)*log(1-p)
    cost = -np.sum(a + b) / p.shape[0]
    return cost

def normalize(matrix):
    '''
    scales the values in a matrix to be in the range [0,1]
    ## Parameters
        - matrix: the matrix to be normalized
    ## Returns
        - out: the normalized matrix
    '''
    col_max = matrix.max(axis=0)
    col_min = matrix.min(axis=0)
    return (matrix - col_min)/(col_max-col_min)

def load_data(filename = 'Project/datasets/wdbc.data'):
    '''
    loads data from filename.csv into a numpy array assuming the first column is sample ID and the second is classification
    ## Parameters
        - filename: path to the csv file
    ## Returns
        - X: numpy array containing features, one row per sample
        - Y: labels corresponding to the samples, one row per sample
    '''
    # Parse CSV into array using numpy. Lambda converts M=>1.0 and B=>0.0
    data = np.genfromtxt(filename, delimiter=',', converters={1: lambda x: 1.0 if x==b'M' else 0.0})
    # Break information into normalized data and labels
    X = normalize(data[:,2:])
    Y = data[:,1:2]
    return X, Y
    load_data()
    np.array()

def partition_data(X, Y, split_percent=0.8):
    '''
    produces randomized subsets of the input matrices X and Y whose relative size depends on the value of split_percent
    ## Parameters
        - X: a numpy array containing features, one sample per row
        - Y: a numpy array containing labelse corresponding to
    '''
    # align smaples on rows [f1, f2, ..., fn, label]
    data = np.hstack((X, Y))
    # randomly reorder samples (rows)
    np.random.shuffle(data)
    # split data into training and testing sets with matching labels
    split_index = int(split_percent*data.shape[0])
    x_train = data[0:split_index,:-1]
    x_test  = data[split_index+1:,:-1]
    y_train = data[0:split_index,-1:]
    y_test  = data[split_index+1:,-1:]
    return x_train, x_test, y_train, y_test

def train(x, y, epochs = 100000, learning_rate=0.1):
    '''
    performs training for a logistic regression classifier to determine the values of a set of weights for each feature and a bias term
    ## Parameters
        - x: numpy array of input features with each row representing one sample
        - y: numpy array of labels with each row representing one sample
        - epochs: the number of training iterations to perform, default=100000
        - learningrate: coefficient for gradient descent, default=0.1
    ## Returns
        - w: numpy array of weights corresponding to the input features
        - b: a bias term added to be added to the weighted sum of features
    '''
    x, y = np.transpose(x), y.reshape(1, y.shape[0])
    w = np.random.randn(x.shape[0], 1)*0.01
    b = 0
    m = x.shape[1]
    loss = []

    for epoch in range(epochs):
        z = np.dot(np.transpose(w), x) + b
        p = sigmoid(z)
        cost = calc_cost(p, y)
        loss.append(np.squeeze(cost))
        dz = p-y
        dw = np.dot(x, np.transpose(dz)) / m
        db = np.sum(dz) / m
        w -= learning_rate * dw 
        b -= learning_rate * db
    plt.figure()
    plt.plot(loss)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cost")
    plt.title("Training Accuracy vs Epochs")
    return w, b

def predict(probabilities, cutoff):
    predictions = np.zeros_like(probabilities)
    for i in range(0, probabilities.shape[1]):
        if probabilities[0][i] >= cutoff:
            predictions[0][i] = 1.0
    return predictions

def test(w, b, x, y):
    '''
    tests accuracy of a trained logistic classifier to produce quality metrics
    ## Parameters
        - w: the weights of the logistic classifier corresponding to the input features
        - b: a bias term added to the weighted sum of input features
        - x: the input features used for classification with each row representing one sample
        - y: the correct classification labels corresponding to the samples of x
    ## Returns
        - accuracy: the percentage of samples correctly classified
        - precision: the percentage of results classified as positive that should be positive
        - recall: the percentage of all positive results that were classified as positive
    '''
    # predict using logistic classifier
    x, y = np.transpose(x), y.reshape(1, y.shape[0])
    z = np.dot(np.transpose(w), x) + b
    p = sigmoid(z)
    p = predict(p, 0.4)

    # count t/f positives and negatives
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(0, p.shape[1]):
        if(y[0,i]==0): # should be false
            if(p[0,i]==0): tn += 1 # predict correctly
            else: fn += 1          # predict incorrectly
        else: # should be true
            if(p[0,i]==1): tp += 1 # predict correctly
            else: fp += 1          # predict incorrectly
        
    # from t/f pos/neg determine metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) # % correctly classified
    precision = tp / (tp + fp) # % of positive results that are correct
    recall = tp / (tp + fn) # % of total positive results that were identified
    return accuracy, precision, recall

def hyper_tune():
    '''
    Performs logistic classifier training and validation for a wide range of epochs and learning rates to determine their optimal value. Outputs of each epoch/learning rate pair are saved to a 'results.csv' file. This function may take significant time (1+ hrs) to run
    '''
    X, Y = load_data()
    x_train, x_test, y_train, y_test = partition_data(X, Y)
    results = []

    x_train, y_train = np.transpose(x_train), y_train.reshape(1, y_train.shape[0])
    w = np.random.randn(x_train.shape[0], 1)*0.01
    b = 0
    m = x_train.shape[1]

    for learning_rate in frange(0.002, 0.3, 0.002): # sweep learning rate 0-0.5 in steps of 0.005
        print("lr:", learning_rate)
        epoch = 1
        while epoch < 100000: # sweep # epocs 0-100,000
            # peform training
            z = np.dot(np.transpose(w), x_train) + b
            p = sigmoid(z)
            cost = calc_cost(p, y_train)
            dz = p-y_train
            dw = np.dot(x_train, np.transpose(dz)) / m
            db = np.sum(dz) / m
            w -= learning_rate * dw 
            b -= learning_rate * db

            # every 100 epochs test and save results
            if(epoch % 100 == 0): 
                accuracy, precision, recall = test(w, b, x_test, y_test)
                results.append([epoch, learning_rate, accuracy, precision, recall])
            
            # increment while loop
            epoch += 1

    # save results
    with open('results.csv', 'at', newline='') as f:
        csv_writer = csv.writer(f)
        header = ["epochs", "lr", "accuracy", "precision", "recall"]
        csv_writer.writerow(header)
        csv_writer.writerows(results)

if __name__ == '__main__':
    print("Logistic Regression Classifier:")
    
    # load and preprocess data
    X, Y = load_data()
    print("Loaded", X.shape[0], "samples")

    # split up data
    x_train, x_test, y_train, y_test = partition_data(X, Y,)
    print(x_train.shape[0], "training samples,", x_test.shape[0], "test samples")

    # train classifier for 50000 epochs and learning_rate=0.05
    w, b = train(x_train, y_train, 50000, 0.05)

    # test classifier and print results
    accuracy, precision, recall = test(w, b, x_test, y_test)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    
    # plot loss over time
    plt.show()