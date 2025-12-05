# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 13:15:58 2025

@author: zhy86
"""

import time
import random
import math
import numpy as np

# Read file

def load_distance_matrix(filename, num_nodes=1000):
    print(f"Loading {filename} ...")
    # Initialize a matrix filled with zeros, Size:1000x1000
    matrix = np.zeros((num_nodes, num_nodes))
    
    count = 0
    with open(filename, 'r', encoding='latin1') as f:
        for line in f:
            parts = line.strip().split()
            
            if len(parts) != 3:
                continue
                
            try:
                # 尝试解析 u, v, w
                u = int(parts[0]) - 1 # Convert to 0-based
                v = int(parts[1]) - 1
                dist = float(parts[2])
                
                matrix[u][v] = dist
                matrix[v][u] = dist
                count += 1
            except ValueError: # Skip text lines
                continue
                
    print(f"Loaded {count} edges into matrix.")
    return matrix

# 2-Opt
def two_opt_fast(route, dist_matrix, start_time, time_limit):

    best_route = route
    num_nodes = len(route) - 1 
    
    # Calculate the initial total distance
    current_dist = 0
    for k in range(num_nodes):
        current_dist += dist_matrix[best_route[k]][best_route[k+1]]
        
    evals = 0
    improved = True
    
    while improved:
        improved = False
        if time.time() - start_time > time_limit: # Timeout Check
            break
            
        for i in range(1, num_nodes - 1):
            for j in range(i + 1, num_nodes):
                evals += 1
                
                # 每 5000 次检查时间 (性能平衡)
                if evals % 5000 == 0:
                    if time.time() - start_time > time_limit:
                        return best_route, current_dist, evals

                # Core Optimization logic
                u = best_route[i-1]
                v = best_route[i]     # 将断开 (u, v)
                x = best_route[j]     # 将断开 (x, y)
                y = best_route[j+1]
                
                # O(1), Perform a local comparison replacing only these 4 edges
                old_cost = dist_matrix[u][v] + dist_matrix[x][y]
                new_cost = dist_matrix[u][x] + dist_matrix[v][y]
                
                if new_cost < old_cost - 1e-9:
                    best_route[i:j+1] = reversed(best_route[i:j+1])
                    current_dist += (new_cost - old_cost)
                    improved = True
                    # G
            
            if improved and (time.time() - start_time > time_limit):
                break
            
    return best_route, current_dist, evals


def get_nearest_neighbor_path(num_nodes, dist_matrix, start_node):
    unvisited = set(range(num_nodes))
    unvisited.remove(start_node)
    path = [start_node]
    curr = start_node
    
    while unvisited:
        # Use numpy argmin or a simple loop to find nearest neighbors
        # For 1000 points, a simple min() is fast enough
        nxt = min(unvisited, key=lambda x: dist_matrix[curr][x])
        unvisited.remove(nxt)
        path.append(nxt)
        curr = nxt
        
    path.append(start_node) # Back to O
    return path

# Main

def solve_tsp(filename, time_limit=58):
    t_start = time.time()
    
    # 
    matrix = load_distance_matrix(filename)
    num_nodes = matrix.shape[0]
    node_ids = list(range(num_nodes))
    
    total_evals = 0
    best_global_cost = float('inf')
    best_global_route = []
    
    # Multi-start
    start_node_idx = 0
    
    print(f"Starting optimization (Time limit: {time_limit}s)...")
    
    while time.time() - t_start < time_limit:
        if start_node_idx < num_nodes:
            start_node = start_node_idx
            start_node_idx += 1
        else:
            start_node = random.randint(0, num_nodes - 1)
            
        # Nearest Neighbor
        initial_route = get_nearest_neighbor_path(num_nodes, matrix, start_node)
        
        # 2-Opt
        route, cost, runs = two_opt_fast(initial_route, matrix, t_start, time_limit)
        total_evals += runs
        
        if cost < best_global_cost:
            best_global_cost = cost
            best_global_route = route[:]
            print(f" >> New Best: {best_global_cost:.2f} (Evals: {total_evals:.1e})")
            
    return best_global_route, best_global_cost, total_evals



# fileA = r"C:\Users\zhy86\Desktop\HW3\TSP_1000_euclidianDistance.txt"
fileB = r"C:\Users\zhy86\Desktop\HW3\TSP_1000_randomDistance.txt"

# Graph A
# print("\n" + "="*30)
# print("Solving Graph A (Euclidean Distance File)...")
# bestA, costA, evalsA = solve_tsp(fileA)

#  Graph B
print("\n" + "="*30)
print("Solving Graph B (Random Distance File)...")
bestB, costB, evalsB = solve_tsp(fileB)


def format_and_save(filename_suffix, route, cost, evals, student_id):

    final_cost = math.ceil(cost * 100) / 100

    final_evals = "{:.1e}".format(evals)
    
    print(f"\nFinal Results for {filename_suffix}:")
    print(f"  Evaluated Cycles: {final_evals}")
    print(f"  Best Cost: {final_cost:.2f}")
    
    # 1-based (1~1000)
    output_route = [x + 1 for x in route]
    
    output_filename = f"solution_{student_id}_{filename_suffix}.txt" 
    with open(output_filename, "w") as f:
        f.write(", ".join(map(str, output_route)))
    print(f"  Saved path to {output_filename}")

SID = "924217254"

#format_and_save("GraphA", bestA, costA, evalsA, SID)
format_and_save("GraphB", bestB, costB, evalsB, SID)