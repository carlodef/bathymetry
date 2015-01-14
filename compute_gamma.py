import numpy as np
from pylab import *

def compute_gamma(depth_map,W,H,A,B):
    r1 = 8
    r2 = 14


    grad = np.gradient(depth_map)
    gradx = grad[0]
    grady = grad[1]

    def cross_prod(X, Y):
        return X[0]*Y[1] - X[1]*Y[0]
    def dist(X, Y):
        return np.sqrt((X[0] - Y[0])**2 + (X[1] - Y[1])**2)

    gamma = list()
    gamma.append((A[0],A[1]))
    iter = 0
    maxit = 100

    while gamma[-1] != B and iter < maxit:
        iter = iter + 1
        curpt = gamma[-1]
        grad_cur = [gradx[curpt[0],curpt[1]], grady[curpt[0],curpt[1]]]
        score = np.zeros((W,H))
        for i in range(W):
            for j in range(H):
                dist_to_cur = dist(curpt, [i,j])
                dist_to_B = dist([i,j], B)
                dist_from_cur_to_B = dist(curpt, B)
                if dist_to_B < dist_from_cur_to_B and dist_to_cur > r1 and dist_to_cur < r2:
                    score[i,j] = abs(cross_prod(grad_cur, [gradx[i,j], grady[i,j]]))
                    # if([i,j] == B or dist([i,j], B) < r2):
                    #     score[i,j] = inf

                else:
                    score[i,j] = -1


        imax,jmax = np.unravel_index(score.argmax(), score.shape)
        if dist(curpt, B) < r2:
            imax,jmax=B[0],B[1]
        gamma.append((imax,jmax))
    gamma.append(B)
    return gamma
