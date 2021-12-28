from this import d
from jwt import ExpiredSignatureError
import networkx as nx
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from IPython.display import Image
from operator import mul
from functools import reduce
import time

function_params = {}
functions = [
    "1", "2", "3", "4", "5", "6"
]
# variable_params = {}
# variables = ["(1,2)", "(1,3)", "(2,3)", "(1,4)", "(2,5)", "(3,6)"]
edges = [
    ("1", "2"),
    ("1", "3"),
    ("2", "3"),
    ("1", "4"),
    ("2", "5"),
    # ("3", "6")
]
# edges = [
#     ("(1,2)", "1"),
#     ("(1,2)", "2"),
#     ("(1,3)", "1"),
#     ("(1,3)", "3"),
#     ("(2,3)", "2"),
#     ("(2,3)", "3"),
#     ("(1,4)", "1"),
#     ("(1,4)", "4"),
#     ("(2,5)", "2"),
#     ("(2,5)", "5"),
#     ("(3,6)", "3"),
#     ("(3,6)", "6")
#     ]

example_graph = nx.Graph()
# example_graph.add_nodes_from([(var, variable_params) for var in variables])
example_graph.add_nodes_from([(var, function_params) for var in functions])
example_graph.add_edges_from(edges)
# fig, axs = plt.subplots()
# nx.draw(example_graph, ax=axs, node_shape="s", with_labels=True, node_color="white")
# # plt.savefig("/Users/philippvonbachmann/Documents/University/WiSe2122/APMLA/APMLA-WS_21-22_material/Assignment/example_graph.jpg")
# plt.show()
# raise ValueError

def compute_marginal(G, var_message):
    marginal=1
    for fac in var_message:
        marginal *= var_message[fac][0]
    return marginal

def convergence(message_new, G, abs_tol):
    edge_marginals = {}
    # soft marginals
    for edge in G.edges():
        edge_marginals[edge] = compute_marginal(G, message_new[edge]) # if p(X=0) > 0.5, choose X=0 else X=1 (by round operation)

    _ = 0
    # hard marginals
    for edge in G.edges():
        edge_marginals[edge] = 1 - round(compute_marginal(G, message_new[edge])) # if p(X=0) > 0.5, choose X=0 else X=1 (by round operation)
    


    err = 0
    n_total_edges = len(G.edges)
    for node in G.nodes():
        node_edges = 0
        for edge in G.edges(node):
            try:
                node_edges += edge_marginals[edge]
            except:
                node_edges += edge_marginals[(edge[1], edge[0])]
        err += int(node_edges > 1)
    err /= n_total_edges
    if err < abs_tol:
        plot_marginal_graph(edge_marginals, G)
    return(err, err < abs_tol)

def plot_marginal_graph(edge_marginals, G):
    new_graph = nx.Graph()
    for node in G.nodes():
        new_graph.add_node(node)
    for edge, value in edge_marginals.items():
        if value==1:
            new_graph.add_edge(edge[0], edge[1])
    fig, axs = plt.subplots()
    nx.draw(new_graph, ax=axs, node_shape="s", with_labels=True)
    plt.show()


def compute_fac_to_var_message(G, fac, var):
    # for computation, we assume variable/egde is 0
    message_sum = 0
    neighbour_edges = G.edges(fac)
    neighbour_edges = list(G.edges(fac))
    neighbour_edges.remove(var)
    for active_edge in neighbour_edges: # we just need to iterate over all possible neighbours since just one of them can be one at a time
        zero_edges = neighbour_edges.copy()
        zero_edges.remove(active_edge)
        product = [G.edges()[edge]['message_t'][fac][1] for edge in zero_edges] # get the incoming message to fac for all neighbours beeing 0
        active_prob = (1 - G[active_edge[0]][active_edge[1]]["message_t"][fac][1])
        message_sum += reduce(mul, product, 1)* active_prob # multiply be incoming for neighbour beeing 1 and add to sum

    return message_sum

def compute_var_to_fac_message(G, fac, var):
    other = var[0] if var[0]!=fac else var[1]
    return G.edges()[var]['message_t'][other][0] # incoming message from other factor

def BP(G, init='random', update='parallel', max_it=1000, abs_tol=1e-4, alpha=0.1, report=False, seed=98):
    if init == 'random':
        np.random.seed(seed)
        for e in G.edges():
            # for each edge we have two messages, one in each direction
            # for both directions we have bernoulli so we could pass the probability distribution as [p(x=0), p(x=1)]
            p = np.random.rand(4)
            # we have order by connection to factor node and the incoming and outgoing message
            # incoming message at pos 0
            G.edges()[e]['message_t'] = {e[0]: [p[0], p[1]], e[1]: [p[2], p[3]]}# FILL just randomly init each message
    elif init == 'all-negative':
        p_value = 0.8
        for e in G.edges():
            G.edges()[e]['message_t'] = {e[0]: [p_value, p_value], e[1]: [p_value, p_value]}# dummy FILL init all mesages as believen sigma_ij is 0 
    elif init == 'all-positive':
        p_value = 0.1
        for e in G.edges():
            G.edges()[e]['message_t'] = {e[0]: [p_value, p_value], e[1]: [p_value, p_value]}# dummy FILL init all messages as believing sigma_ij is 1 
        


    # Iterating
    conv, it = False, 0# FILL
    differences = []
    
    if update=='parallel':
        while not conv and (it < 10000): # FILL
            print("iteration", it)
            for e in G.edges():
                print(e, G.edges()[e]['message_t'])
            message_new = {}
            for e in G.edges():
                i = e[0]
                j = e[1]
                new_message_t = {i:[None, None], j: [None, None]}
                # first update incoming messages
                new_message_t[i][0] = compute_fac_to_var_message(G, i, (i,j))
                new_message_t[j][0] = compute_fac_to_var_message(G, j, (j,i))
                # then update outgoing messages
                new_message_t[i][1] = compute_var_to_fac_message(G, i, (i,j))
                new_message_t[j][1] = compute_var_to_fac_message(G, j, (j,i))
                message_new[e] = new_message_t
                # FILL
                # here update both messages as in the equations, however dont save directly since we only should update after timestep   
                # FILL
                    
            diff, conv = convergence(message_new, G, abs_tol)
            differences.append(diff)
            it+=1
            # FILL
            # now update the whole edges
            for e in G.edges():
                G.edges()[e]['message_t'] = message_new[e]

    elif update=='random':
        while not conv and (it < 1000):# FILL:
            print("iteration", it)
            for e in G.edges():
                print(e, G.edges()[e]['message_t'])
            message_old = {}
            perm = list(G.edges())
            np.random.seed(seed)
            np.random.shuffle(perm)
            for e in perm:
                # message_old[e] = -1# FILL why do we need to fill this??
                i = e[0]
                j = e[1]
                new_message_t = {i:[None, None], j: [None, None]}
                # first update incoming messages
                new_message_t[i][0] = compute_fac_to_var_message(G, i, (i,j))
                new_message_t[j][0] = compute_fac_to_var_message(G, j, (j,i))
                # then update outgoing messages
                new_message_t[i][1] = compute_var_to_fac_message(G, i, (i,j))
                new_message_t[j][1] = compute_var_to_fac_message(G, j, (j,i))
                # here we directly update instead of first computing all updates and then looking at them
                G.edges()[e]['message_t'] = new_message_t
                message_old[e] = new_message_t

            diff, conv = convergence(message_old, G, abs_tol)
            differences.append(diff)
            it+=1
            seed+=1
    
    if report:
        print('Number of iterations: {0}'.format(it))
    
    return(it, differences)

BP(example_graph, update="random", seed=10, init="all-negative")

for e in example_graph.edges():
    print(example_graph.edges()[e])