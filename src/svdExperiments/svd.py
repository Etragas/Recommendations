import utils
import numpy as np
from dl import *
from scipy import linalg
from numpy import dot

full_data = dl().LoadData(file_path="../Data/ml-100k/u.data", data_type=dl.MOVIELENS, size= (1200, 2000))
# Reduce the matrix to toy size
#full_data = full_data[:100,:100]
rows = [x for x in range((full_data.shape[0])) if full_data[x,:].sum() > 0]
cols = [x for x in range((full_data.shape[1])) if full_data[:,x].sum() > 0]
full_data = full_data[rows,:]
full_data = full_data[:,cols]

def nmf(X, latent_features, max_iter=1000, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-3
    print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
    #X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i == max_iter or i == 1 or i % 5 == 0:
            print 'Iteration {}:'.format(i),
            X_est = np.round(dot(A, Y))
            X_est[X_est > 5] = 5
            X_est[X_est < 0] = 0
            print X_est
            print full_data
            err = np.round(X_est) - full_data
            loss = np.sqrt(np.sum(err ** 2))

            print 'rmse', np.round(loss, 4),
            if loss < fit_error_limit:
                break

		return A, Y

nmf(full_data, 40, max_iter=1000)
