import scipy as sp
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import *

from matplotlib import pyplot as plt

# Parameters
N = 35
alpha = 1.0
beta = 0.0
L = 1.0
hx,hy = L/N, L/N

# Depth
h=lambda u,v: (u-0.2)**2 + (v-0.7)**2

# matrice 
G = csr_matrix((N*N,N*N))

ex = np.ones(N);
lp1 = sp.sparse.spdiags(np.vstack((ex,  0*ex, ex)), [-1, 0, 1], N, N, 'csr'); 
e = sp.sparse.eye(N)
G = sp.sparse.kron(lp1, e) + sp.sparse.kron(e, lp1)

for j in range(N-1):
  for i in range(N-1):
    #k = i+j*N
    G[i+j*N, (i+1)+j*N] = alpha*hx*hx + beta*(h(i*hx , j*hy )  - h((i+1)*hx,j*hy))**2
    G[i+j*N, i+(j+1)*N] = alpha*hy*hy + beta*(h(i*hx , j*hy )  - h(i*hx,(j+1)*hy))**2
    G[(i+1)+j*N, i+j*N] = G[i+j*N, (i+1)+j*N]
    G[i+(j+1)*N, i+j*N] = G[i+j*N, i+(j+1)*N]


dist_matrix, pred = dijkstra( G, return_predecessors=True )

xstart,ystart = 0,0
start = xstart+ystart*N
xend,yend = N-12,N-1
end = xend+yend*N

print dist_matrix[start,end] 

cur = end
path = np.zeros((N*2,2))
i = 0
while cur != start:
  
  cur = pred[start, cur]
  print cur
  path[i,0] = cur%N *hx 
  path[i,1] = cur/N * hy
  i += 1
  
coo = np.array([xstart*hx,ystart*hy])


plt.figure(1)
plt.plot(path[:,0],path[:,1],'-db')
plt.plot(path[0,0],path[0,1],'dr')
plt.show()
