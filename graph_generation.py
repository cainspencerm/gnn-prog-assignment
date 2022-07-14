import numpy as np
from tqdm import tqdm
import mnist_mini
import networkx as nx
import os


# Modified from source: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

def knn(data, query, query_idx, k, distance_fn):
    neighbor_distances_and_indices = []
    
    # For each example in the data
    for idx, (example, _) in enumerate(data):
        if idx == query_idx:
            continue
        example = example.numpy().flatten()
        # Calculate the distance between the query example and the current
        # example from the data.
        distance = distance_fn(example, query)
        
        # Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, idx))
    
    # Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    
    # Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    return k_nearest_distances_and_indices


dataset = mnist_mini.MNIST('./data', split='full')

edge_list = []
if os.path.exists('edge_list.txt'):
    with open('edge_list.txt', 'r') as f:
        for line in f.readlines():
            edge_list.append(tuple(map(int, line.split())))

else:
    t = tqdm(total=len(dataset))
    for idx, (image, _) in enumerate(dataset):
        results = knn(dataset, image.numpy().flatten(), idx, k=7, distance_fn=np.correlate)
        for result in results:
            edge_list.append((idx, result[1]))
        t.update(1)
    t.close()

    with open('edge_list.txt', 'w') as f:
        for node1, node2 in edge_list:
            f.write(f'{node1} {node2}\n')

full_adj = np.zeros((len(dataset), len(dataset)))
for node1, node2 in edge_list:
    full_adj[node1, node2] = 1
    full_adj[node2, node1] = 1

super_adj = np.zeros((10, 10))
for idx, (_, i_label) in enumerate(dataset):
    for jdx, (_, j_label) in enumerate(dataset):
        if idx == jdx:
            continue
        super_adj[i_label, j_label] += 1

print(super_adj)