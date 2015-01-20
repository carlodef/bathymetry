import numpy as np
import matplotlib.pylab as plt
import scipy.sparse

import cost
import plot_sets


def select_vertices(im, n, thresh):
    """
    """
    gx, gy = np.gradient(im)
    gnorm = gx*gx + gy*gy
    out = list()

    # select greatest gradients on each n x n subgrid
    rows = np.linspace(0, np.shape(im)[0]+1, n).astype('int')
    cols = np.linspace(0, np.shape(im)[1]+1, n).astype('int')
    for i in range(len(rows)-1):
        for j in range(len(cols)-1):
                    sub_im_grad = gnorm[rows[i]:rows[i+1], cols[j]:cols[j+1]]
                    l = select_vertices_in_small_image(sub_im_grad, thresh, rows[i], cols[j])
                    out += l
    return out


def select_vertices_in_small_image(gradient_image, thresh, offset_row=0, offset_col=0):
    """
    """
    m = np.max(gradient_image)
    row, col = np.where(gradient_image >= m * (1 - thresh))
    row = np.array(row)
    col = np.array(col)
    row += offset_row
    col += offset_col
    return [(row[i], col[i]) for i in range(len(row))]


def distances_matrix(vertices):
    n = len(vertices)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            p = vertices[i]
            q = vertices[j]
            d = np.sqrt((p[0] - q[0])**2 + (p[1] - q[1])**2)
            out[i, j] = d
            out[j, i] = d
    return out


def build_edges_in_graphs(vertices, dmin, dmax, im, eps_v, eps_h):
    """
    Args:
        vertices: list of tuples. Each tuple has length 2, and contains
            the coordinates of a pixel.
        dmin, dmax: min and max distance between two connected vertices
        im: depth map
        eps_v, eps_h: thresholds

    Returns:
        sparse matrix
    """
    m = distances_matrix(vertices)
    rows, cols = np.where((m > dmin) * (m < dmax))
    rows_sparse = np.zeros(2*len(rows))
    cols_sparse = np.zeros(2*len(rows))
    vals_sparse = np.zeros(2*len(rows))
    for i in range(len(rows)):
        a = np.array(vertices[rows[i]])
        b = np.array(vertices[cols[i]])
        c = cost.cost_straight_path(im, a, b, eps_v, eps_h)[0]
        rows_sparse[2*i] = rows[i]
        cols_sparse[2*i] = cols[i]
        vals_sparse[2*i] = c
        rows_sparse[2*i + 1] = cols[i]
        cols_sparse[2*i + 1] = rows[i]
        vals_sparse[2*i + 1] = c
    return scipy.sparse.csr_matrix((vals_sparse, (rows_sparse, cols_sparse)))


def find_shortest_path(m, vertices):
    start = m.shape[0] - 2
    end = m.shape[0] - 1
    dist_matrix, pred = scipy.sparse.csgraph.dijkstra(m, return_predecessors=True, indices=[start])

    cur = end
    path = list()
    while cur != start and cur != -9999:
        path.append(vertices[cur])
        cur = pred[0, cur]
    if (cur== start):
	path.append(vertices[cur])
    path.reverse()
    return path


def main(A=(10, 17), B=(91, 77), plots_dir='sd', filename="morne_rouge.asc",
        w=101, h=101, n_grid=10, thresh=0.3, rmin=8, rmax=12, eps_v=1,
        eps_h=0.1):
    """
    """
    depth_map = cost.load_image(filename, w, h)

    # compute the shortest path
    vertices = select_vertices(depth_map, n_grid, thresh) 
    vertices.append(A)
    vertices.append(B)
    graph = build_edges_in_graphs(vertices, rmin, rmax, depth_map, eps_v, eps_h)
    path = find_shortest_path(graph, vertices)  
    c, J_record = cost.cost(depth_map, path, eps_v, eps_h)
    print "the optimal path is ", path
    print "the optimal cost is ", c

    # plot it 
    plot_sets.main(filename, w, h, N=len(path), gamma=path, eps_v=eps_v, eps_h=eps_h, figures_path=plots_dir)


def show_points(im, points):
    """
    """
    im_to_show = im.copy()
    for p in points:
        im_to_show[p[0], p[1]] = np.nan
    plt.imshow(im_to_show, interpolation='nearest')
    plt.show()


if __name__ == '__main__':
    main()
