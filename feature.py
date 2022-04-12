import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset
    
    returns:
        X_new - (N, d + num_new_features) array with 
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    
    #TODO: implement this
    N = X.shape[0]
    d = X.shape[1]
    X_new = np.zeros((N, d + 1))
    X_new[:, :-1] = X

    for i in range(N):
        if (X_new[i][0] <= 1 and X_new[i][1] <= 1) or (X_new[i][0] > 1 and X_new[i][1] > 1):
            X_new[i][2] = -1
        else:
            X_new[i][2] = 1
    return X_new


    
