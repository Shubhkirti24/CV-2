import sys
import numpy as np
import networkx as nx



def read_bribes(filename):
    """Reads the bribe file and returns a 2D array of values."""
    with open(filename, 'r') as f:
        return [[int(x) for x in line.strip().split()] for line in f]




def build_graph(n):
    """Builds a graph representing the grid of houses."""
    G = nx.grid_2d_graph(n, n)
    # for i in range(n):
    #     G.remove_node((i, -1))
    #     G.remove_node((i, n))
    #     G.remove_node((-1, i))
    #     G.remove_node((n, i))
    return G



def build_potentials(n, bribes_r, bribes_d):
    """Builds the unary and pairwise potentials for the MRF."""
    pot_u = np.zeros((n, n, 2))
    pot_u[:, :, 0] = bribes_r
    pot_u[:, :, 1] = bribes_d
    pot_p = np.zeros((2, 2))
    pot_p[0, 1] = pot_p[1, 0] = 1000
    return pot_u, pot_p


def compute_messages(G, pot_u, pot_p, max_iters=1000, tol=1e-6):
    """Runs loopy belief propagation to compute the marginal distributions."""
    n = int(np.sqrt(len(G)))
    # Initialize messages
    msgs_u = {node: np.zeros((2,)) for node in G.nodes}
    msgs_p = {(node, neighbor): np.zeros((2, 2)) for node in G.nodes for neighbor in G.neighbors(node)}
    for node in G.nodes:
        msgs_u[node] = pot_u[node]
    # Run loopy belief propagation
    for _ in range(max_iters):
        msgs_u_prev = msgs_u.copy()
        msgs_p_prev = msgs_p.copy()
        for node in G.nodes:
            # Compute the product of all incoming messages
            prod_p = np.ones((2,))
            for neighbor in G.neighbors(node):
                prod_p *= np.sum(msgs_p[(node, neighbor)], axis=1)
            # Compute the message to send to each neighbor
            for neighbor in G.neighbors(node):
                denominator = np.sum(msgs_p[(node, neighbor)], axis=1)
                if np.any(denominator):
                    prod_p_neighbor = prod_p / denominator
                else:
                    prod_p_neighbor = np.zeros_like(prod_p)
                msgs_u[node] = pot_u[node] * prod_p_neighbor
                msgs_p[(node, neighbor)] = np.ones((2, 2))
                for other_neighbor in G.neighbors(node):
                    if other_neighbor != neighbor:
                        msgs_p[(node, neighbor)] *= msgs_u_prev[other_neighbor]
                msgs_p[(node, neighbor)] *= prod_p_neighbor
        # Check for convergence
        converged = True
        for node in G.nodes:
            if not np.allclose(msgs_u[node], msgs_u_prev[node], atol=tol):
                converged = False
                break
        for edge in G.edges:
            if not np.allclose(msgs_p[edge], msgs_p_prev[edge], atol=tol):
                converged = False
                break
        if converged:
            break
    # Compute the marginal distributions
    marginals = np.zeros((n, n, 2))
    for node in G.nodes:
        marginals[node] = pot_u[node] * np.prod([np.sum(msgs_p[(node, neighbor)], axis=1) for neighbor in G.neighbors(node)], axis=0) * np.prod([msgs_u[neighbor] for neighbor in G.neighbors(node)], axis=0)
    return marginals

if __name__ == '__main__':
    # Read command line arguments
    n = int(sys.argv[1])
    r_file = sys.argv[2]
    d_file = sys.argv[3]
    

    # Read bribe files
    bribes_r = read_bribes(r_file)
    bribes_d = read_bribes(d_file)

    # bribes_r =np.array([[0, 0, 0, 0, 0],[0,10000, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0 ,10, 0, 0],[10000, 0, 0, 0, 0]])
    # bribes_d = np.array([[0, 0, 0, 0, 0],[0, 0, 0, 0, 0],[0, 0, 10000, 0, 0],[500, 0 ,0, 0, 0],[0, 0, 0, 0, 0]])

    # Build graph and potentials
    G = build_graph(n)
    pot_u, pot_p = build_potentials(n, bribes_r, bribes_d)
    
    # Compute marginal distributions
    marginals = compute_messages(G, pot_u, pot_p)
    
    # Find cheapest labeling
    labels = np.argmax(marginals, axis=2)
    cost = np.sum(pot_u[np.arange(n)[:, np.newaxis], np.arange(n), labels])
    print('Computing optimal labeling:')
    for row in labels:
        print(' '.join(['R' if x == 0 else 'D' for x in row]))
    print(f'Total cost = {cost}')
