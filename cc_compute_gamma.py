import numpy as np
from pylab import *
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import *
from scipy.io import savemat

def compute_gamma(depth_map,W,H,A,B,alpha):
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

    #score = np.zeros((2*r2-1,2*r2-1))

    assert(W==H)
    N = W
    G = lil_matrix((N*N,N*N))
    for m in range(N):
	for n in range(N):
	    curpt = [m,n]
	    #print curpt
	    grad_cur = [gradx[curpt[0],curpt[1]], grady[curpt[0],curpt[1]]]
	    #for i in range(max(0, curpt[0]-r2), min(W, curpt[0]+r2+1)):
	    for k in range(r2):
		i = curpt[0] + k
		for l in range(r2):
		    j = curpt[1] + l
		    dist_to_cur = dist(curpt, [i,j])
		    dist_to_B = dist([i,j], B)
		    dist_from_cur_to_B = dist(curpt, B)
		    dist_gain = dist_to_B - dist_from_cur_to_B 
		    if i < W and j < H and dist_to_B < dist_from_cur_to_B and dist_to_cur > r1 and dist_to_cur < r2:
		    #if i < W and j < H and  dist_to_cur > r1 and dist_to_cur < r2:
			this_score = 1./(abs(cross_prod(grad_cur, [gradx[i,j], grady[i,j]])) - dist_gain*alpha)
			G[m+n*N,i+j*N] = this_score
			G[i+j*N,m+n*N] = this_score
			# if([i,j] == B or dist([i,j], B) < r2):
			#     score[i,j] = inf

		    #else:
			#score[k,l] = -1


    #savemat('prova', {'G':G})
    print 'Dijkstra start'

    xstart,ystart = A[0],A[1]
    start = xstart+ystart*N
    xend,yend = B[0],B[1]
    end = xend+yend*N

    dist_matrix, pred = dijkstra( G, return_predecessors=True , indices = start)
    #print dist_matrix[start,end] 

    cur = end
    path = np.array([0,0])
    i = 0
    #print pred
    gamma.append((xend, yend))
    while cur != start:
	#print cur
	#cur = pred[start, cur]
	cur = pred[ cur]
	i = cur%N 
	j = cur/N
	gamma.append((i,j))
      #i += 1

    gamma.reverse()
      

        #imax,jmax = np.unravel_index(score.argmax(), score.shape)
	#imax,jmax = imax+curpt[0]-r2+1,jmax+curpt[1]-r2+1
        #if dist(curpt, B) < r2:
            #imax,jmax=B[0],B[1]
        #gamma.append((imax,jmax))
    #gamma.append(B)

    print gamma
    return gamma
