import baseline as a
import greedy as b
import optimization_gradient_and_dijkstra as c
import cc_anneaux as d

a.main(A=(10,17), B=(91, 77), plots_dir='baseline')
b.main(A=(10,17), B=(91, 77), plots_dir='greedy')
c.main(A=(10,17), B=(91, 77), plots_dir='gradient_dijkstra')
d.main(A=(10,17), B=(91, 77), plots_dir='anneaux')
