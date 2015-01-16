import numpy as np
from pylab import *

def compute_gamma(depth_map,W,H,A,B):
    r1 = 8
    r2 = 12


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

    score = np.zeros((2*r2-1,2*r2-1))
    while gamma[-1] != B and iter < maxit:
        iter = iter + 1
        curpt = gamma[-1]
        grad_cur = [gradx[curpt[0],curpt[1]], grady[curpt[0],curpt[1]]]
        #for i in range(max(0, curpt[0]-r2), min(W, curpt[0]+r2+1)):
        for k in range(2*r2-1):
	    i = curpt[0] + k - r2 + 1
	    for l in range(2*r2-1):
		j = curpt[1]+ l - r2 + 1
                dist_to_cur = dist(curpt, [i,j])
                dist_to_B = dist([i,j], B)
                dist_from_cur_to_B = dist(curpt, B)
		dist_gain = dist_to_B - dist_from_cur_to_B 
		if dist_to_B < dist_from_cur_to_B and dist_to_cur > r1 and dist_to_cur < r2:
                    score[k,l] = abs(cross_prod(grad_cur, [gradx[i,j], grady[i,j]])) - dist_gain*0.01000
                    # if([i,j] == B or dist([i,j], B) < r2):
                    #     score[i,j] = inf

                else:
                    score[k,l] = -1


        imax,jmax = np.unravel_index(score.argmax(), score.shape)
	imax,jmax = imax+curpt[0]-r2+1,jmax+curpt[1]-r2+1
        if dist(curpt, B) < r2:
            imax,jmax=B[0],B[1]
        gamma.append((imax,jmax))
    gamma.append(B)
    return gamma
