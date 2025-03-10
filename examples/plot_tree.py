import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def plot_rrt_star_tsne(root, all_nodes, last_added_node=None, path_nodes=None):
    if len(all_nodes) < 3:
        return None
    # Extract states and costs
    states = np.array([node.config for node in all_nodes])
    costs = np.array([node.cost for node in all_nodes])
    
    
    # Normalize costs for coloring
    min_cost, max_cost = np.min(costs), np.max(costs)
    norm_costs = (costs - min_cost) / (max_cost - min_cost)  # Normalize to [0, 1]
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    states_2d = pca.fit_transform(states)

    # Create a mapping from nodes to their projected 2D positions
    node_to_2d = {node: pos for node, pos in zip(all_nodes, states_2d)}

    # Plot edges (tree structure)
    plt.figure(figsize=(8, 8))
    for node in all_nodes:
        if node.parent is not None:
            parent_pos = node_to_2d[node.parent]
            node_pos = node_to_2d[node]
            plt.plot([parent_pos[0], node_pos[0]], [parent_pos[1], node_pos[1]], "gray", alpha=0.5)

    # Plot nodes with cost-based coloring
    scatter = plt.scatter(states_2d[:, 0], states_2d[:, 1], c=norm_costs, cmap="viridis", s=20)
    plt.colorbar(scatter, label="Normalized Cost-to-Come")

    # Highlight root (start node)
    start_pos = node_to_2d[root]
    plt.scatter(start_pos[0], start_pos[1], c="red", s=100, marker="*", label="Start Node")

    # Highlight last added node
    if last_added_node is not None:
        # print("Adding last node")
        last_node_pos = node_to_2d[last_added_node]  # Last node in the list
        plt.scatter(last_node_pos[0], last_node_pos[1], s=100, marker="x", label="Last Added Node")


    # Highligth the path if provided
    if path_nodes is not None:
        # breakpoint()
        # Highlight goal node
        goal_pos = node_to_2d[path_nodes[-1]]
        plt.scatter(goal_pos[0], goal_pos[1], c="green", s=100, marker="*", label="Goal Node")

        for i in range(len(path_nodes) - 1):
            parent_pos = node_to_2d[path_nodes[i]]
            node_pos = node_to_2d[path_nodes[i + 1]]
            plt.plot([parent_pos[0], node_pos[0]], [parent_pos[1], node_pos[1]], "yellow", alpha=0.5)

    # add ids
    for node, (x, y) in node_to_2d.items():
        plt.text(x, y, node.name, fontsize=6, ha="center", va="center", color="black")


    plt.legend()
    plt.title("t-SNE Projection of RRT*-Connect Tree with Cost Coloring")
    plt.grid()
    return plt
    

# Example usage:
# plot_rrt_star_tsne(root, all_nodes)
