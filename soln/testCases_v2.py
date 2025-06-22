import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def sigmoid_derivative(z):
    s = 1 / (1 + np.exp(-z))
    ds = s * (1 - s)
    return ds

def forward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.array([[1, 1, 0]])
    parameters = {'W1': np.random.randn(4, 2), 'b1': np.random.randn(4, 1), 
                  'W2': np.random.randn(1, 4), 'b2': np.random.randn(1, 1)}
    return X_assess, Y_assess, parameters

def backward_propagation_test_case():
    np.random.seed(1)
    X_assess = np.random.randn(2, 3)
    Y_assess = np.array([[1, 0, 1]])
    parameters = {'W1': np.random.randn(4, 2), 'b1': np.random.randn(4, 1), 
                  'W2': np.random.randn(1, 4), 'b2': np.random.randn(1, 1)}
    cache = {'Z1': np.random.randn(4, 3), 'A1': np.random.randn(4, 3), 
             'Z2': np.random.randn(1, 3), 'A2': np.random.randn(1, 3)}
    return X_assess, Y_assess, cache, parameters

def update_parameters_test_case():
    parameters = {'W1': np.array([[1.76405235, 0.40015721],
                                  [0.97873798, 2.2408932 ],
                                  [1.86755799, -0.97727788],
                                  [0.95008842, -0.15135721]]),
                  'b1': np.array([[0.],
                                  [0.],
                                  [0.],
                                  [0.]]),
                  'W2': np.array([[0.4105985 , 0.14404357, 1.45427351, 0.76103773]]),
                  'b2': np.array([[0.]])}

    grads = {'dW1': np.array([[ 1.86755799, -0.97727788],
                              [ 0.95008842, -0.15135721],
                              [-0.10321885, 0.4105985 ],
                              [ 0.14404357, 1.45427351]]),
             'db1': np.array([[0.76103773],
                              [0.12167502],
                              [0.44386323],
                              [0.33367433]]),
             'dW2': np.array([[ 1.49407907, -0.20515826, 0.3130677 , -0.85409574]]),
             'db2': np.array([[-2.55298982]])}

    return parameters, grads
