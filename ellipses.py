from pylab import *
from scipy import *
from matplotlib import animation
import scipy.integrate

def plot_matrix(A) :
    w, v = np.linalg.eigh(A)
    print w, v
    for i in (0, 1):
        plot([-w[i]*v[0, i],w[i]*v[0, i]], [-w[i]*v[1, i], w[i]*v[1, i]])
    xlim(-1.0,1.0)
    ylim(-1.0,1.0)
    

G = [[.2,0],[0,0]]
A0 = [[1,.5], [.5, 1]]
A0 = randn(2,2)
# A0 = [[0,0], [0, 0]]
fig = plt.figure()
plot_matrix(A0)
show()
fig = plt.figure()
plot_matrix(G)
show()

def myode_integrate(A0, G, tfinal, Npoints, alpha, beta):
    def fun(y, t):
        A = reshape(y, (2,2))
        return reshape(alpha * eye(2,2) - beta * dot(G,A) - beta * dot(A, G), 4)
    As = scipy.integrate.odeint(fun, reshape(A0, 4), np.linspace(0, tfinal, Npoints))
    return As

Npoints = 100
tfinal = 10
alpha = .1
beta = 1
As = myode_integrate(A0, G, tfinal, Npoints, alpha, beta)


fig = plt.figure()
L = 2
ax = plt.axes(xlim=(-L,L), ylim=(-L, L))
plot1, plot2 = ax.plot([], [], 'r', [], [], 'b')
def init():
    plot1.set_data([], [])
    plot2.set_data([], [])
    return [plot1,plot2]

def animate(i):
    A = reshape(As[i, :], (2,2))
    w, v = np.linalg.eig(A)
    # print w, v
    plot1.set_data([-w[0]*v[0,0],w[0]*v[0,0]], [-w[0]*v[1,0], w[0]*v[1,0]])
    plot2.set_data([-w[1]*v[0,1],w[1]*v[0,1]], [-w[1]*v[1,1], w[1]*v[1,1]])
    # for i in (0, 1):
    #     plot([0,w[i]*v[0, i]], [0, w[i]*v[1, i]])
    # plot1.set_data
    return [plot1, plot2]
    
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=Npoints, interval=20, blit=False)
