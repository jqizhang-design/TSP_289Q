# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 15:29:05 2025

@author: zhy86
"""

import networkx as nx
import numpy as np
import time
import math

def load_distance_matrix(filename, num_nodes=1000):
    print(f"Loading {filename} ...")
    matrix = np.zeros((num_nodes, num_nodes))
    with open(filename, 'r', encoding='latin1') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3: continue
            try:
                u = int(parts[0]) - 1 
                v = int(parts[1]) - 1
                dist = float(parts[2])
                matrix[u][v] = dist
                matrix[v][u] = dist
            except ValueError:
                continue
    return matrix

# Christofides
def run_christofides(matrix):
    print("Running Christofides Algorithm (this may take 5-10 seconds)...")
    num_nodes = matrix.shape[0]
    
    # 1
    G = nx.Graph()
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            G.add_edge(i, j, weight=matrix[i][j])
            
    # 2
    T = nx.minimum_spanning_tree(G, weight='weight')
    # 3
    odd_degree_nodes = [v for v, d in T.degree() if d % 2 == 1]
    
    # 4
    subgraph = G.subgraph(odd_degree_nodes)
    matching = nx.min_weight_matching(subgraph, weight='weight')
    
    # 5
    M = nx.MultiGraph()
    M.add_edges_from(T.edges(data=True))
    M.add_edges_from(matching) # matching 是边的集合 {(u,v), ...}
    
    # 6
    euler_circuit = list(nx.eulerian_circuit(M, source=0))
    
    # 7
    visited = [False] * num_nodes
    route = []
    
    for u, v in euler_circuit:
        if not visited[u]:
            visited[u] = True
            route.append(u)

    # 补全最后一步闭环
    route.append(route[0])
    
    cost = 0
    for k in range(len(route)-1):
        cost += matrix[route[k]][route[k+1]]
        
    print(f"Christofides Initial Cost: {cost:.2f}")
    return route, cost

# Same: 2-Opt
def two_opt_fast(route, dist_matrix, start_time, time_limit):
    best_route = route
    num_nodes = len(route) - 1 
    current_dist = 0
    for k in range(num_nodes):
        current_dist += dist_matrix[best_route[k]][best_route[k+1]]
    evals = 0
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > time_limit: break
        for i in range(1, num_nodes - 1):
            for j in range(i + 1, num_nodes):
                evals += 1
                if evals % 5000 == 0:
                    if time.time() - start_time > time_limit:
                        return best_route, current_dist, evals
                u, v = best_route[i-1], best_route[i]
                x, y = best_route[j], best_route[j+1]
                old_cost = dist_matrix[u][v] + dist_matrix[x][y]
                new_cost = dist_matrix[u][x] + dist_matrix[v][y]
                if new_cost < old_cost - 1e-9:
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    current_dist += (new_cost - old_cost)
                    improved = True
            if improved and (time.time() - start_time > time_limit): break
    return best_route, current_dist, evals

# Main
def solve_with_christofides(filename, time_limit=58):
    t_start = time.time()
       # 1
    matrix = load_distance_matrix(filename)
    
    # 2

    initial_route, initial_cost = run_christofides(matrix)
    
    # 3
    print("Improving with 2-Opt...")
    best_route, best_cost, evals = two_opt_fast(initial_route, matrix, t_start, time_limit)
    
    # 4
    cost_rounded = math.ceil(best_cost * 100) / 100
    print("-" * 30)
    print(f"Final Results (Christofides + 2-Opt):")
    print(f"  Total Evaluations: {evals:.1e}")
    print(f"  Final Cost: {cost_rounded:.2f}")
    return best_route

fileA = r"C:\Users\zhy86\Desktop\HW3\TSP_1000_euclidianDistance.txt"
final_route_A = solve_with_christofides(fileA)
SID = "924217254"
output_filename = f"solution_{SID}_GraphA_Christofides.txt"

with open(output_filename, "w") as f:
    output_string = ", ".join([str(node + 1) for node in final_route_A])
    f.write(output_string)

print(f"\n[Saved] Graph A result saved to: {output_filename}")