import numpy as np
from tqdm import tqdm
from data import mnist_mini
from torch.utils import data
import os
import networkx as nx
from matplotlib import pyplot as plt


# Modified from source: https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

def knn(dataset, query, query_idx, k, distance_fn):
    neighbor_distances_and_indices = []
    
    # For each example in the data
    for idx, (example, _) in enumerate(dataset):
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
            if (idx, result[1]) not in edge_list and (result[1], idx) not in edge_list:
                edge_list.append((idx, result[1]))
        t.update(1)
    t.close()

    with open('edge_list.txt', 'w') as f:
        for node1, node2 in edge_list:
            f.write(f'{node1} {node2}\n')

super_adj = np.zeros((10, 10), dtype=int)
for edge in edge_list:
    _, label1 = dataset[edge[0]]
    _, label2 = dataset[edge[1]]
    super_adj[label1, label2] += 1

for row in super_adj:
    print(' & '.join(map(str, row)), end=' \\\\\n')
print(np.sum(super_adj))

G = nx.Graph()
G.add_edges_from(edge_list)

labels = [dataset[idx][1].item() for idx in range(len(dataset))]
nx.draw(G, node_size=2, nodelist=[i for i in range(len(labels))], node_color=labels, cmap=plt.get_cmap('gist_rainbow'), width=0.5, with_labels=False)
plt.show()