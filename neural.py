# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# %%
data = pd.read_csv('data/train.csv')

# %%
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

# %%
testData = data[0:1000].T
testX = testData[1:n] / 255.0
testY = testData[0]

trainData = data[1000:m].T
trainX = trainData[1:n] / 255.0
trainY = trainData[0]

# %%
def init_params():
    W1 = np.random.randn(128, 784) * np.sqrt(2.0 / 784)
    B1 = np.zeros((128, 1))
    W2 = np.random.randn(10, 128) * np.sqrt(2.0 / 128)
    B2 = np.zeros((10, 1))
    return W1, B1, W2, B2

def ReLU(x):
    return np.maximum(0, x)

def deriv_ReLU(Z):
    return Z > 0

def softmax(x):
    shift_x = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

def one_hot(x):
    one_hot_x = np.zeros((x.size, x.max() + 1))
    one_hot_x[np.arange(x.size), x] = 1
    return one_hot_x.T

def forward_prop(W1, B1, W2, B2, X):
    Z1 = W1.dot(X) + B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + B2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = (1 / m) * dZ1.dot(X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2

def update_params(W1, B1, W2, B2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2
    B1 = B1 - alpha * db1
    B2 = B2 - alpha * db2
    return W1, B1, W2, B2

# %%
def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(pred, Y):
    return np.sum(pred == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    W1, B1, W2, B2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, B1, W2, B2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)
        W1, B1, W2, B2 = update_params(W1, B1, W2, B2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration:", i, "| Accuracy:", get_accuracy(get_predictions(A2), Y))
    return W1, B1, W2, B2

# %%
W1, B1, W2, B2 = gradient_descent(trainX, trainY, 500, 0.1)

# %%

def make_predictions(X, W1, B1, W2, B2):
    _, _, _, A2 = forward_prop(W1, B1, W2, B2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, B1, W2, B2):
    current_image = trainX[:, index, None]
    prediction = make_predictions(trainX[:, index, None], W1, B1, W2, B2)
    label = trainY[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
# %%

test_prediction(0, W1, B1, W2, B2)
test_prediction(1, W1, B1, W2, B2)
test_prediction(2, W1, B1, W2, B2)
test_prediction(3, W1, B1, W2, B2)
test_prediction(67, W1, B1, W2, B2)
test_prediction(69, W1, B1, W2, B2)
test_prediction(117, W1, B1, W2, B2)
test_prediction(108, W1, B1, W2, B2)
# %%
