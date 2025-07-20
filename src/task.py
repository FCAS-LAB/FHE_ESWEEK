import random
import networkx as nx
import matplotlib.pyplot as plt

import read_file

def generate_random_task_graph(num_nodes):
    node_weight = [0]
    for i in range(1, 1 + num_nodes):
        node_weight.append(random.randint(1, 10))
    edge = {i: [j for j in range(1, i) if random.random() < 0.25] for i in range(1, 1 + num_nodes)}  # edge[i] = [j1,j2]; j < i
    edge_weight = {(j, i): random.randint(1, 5) for i in range(1, 1 + num_nodes) for j in edge[i]}

    return node_weight, edge, edge_weight

# node, edge , node_weight, edge_weight -> nx.DiGraph
def create_task_graph(node_weight, edge, edge_wieght):
    task_graph = nx.DiGraph()

    for i, weight in enumerate(node_weight):  # enumerate(n): idx, value
        task_graph.add_node(i, weight=weight)

    for node, dependencies in edge.items():  # key-value.items()
        for dep_node in dependencies:
            task_graph.add_edge(dep_node, node, weight=edge_wieght.get((dep_node, node)))

    return task_graph

def draw_task_graph_1(num_nodes):
    node_weight, edge, edge_weight = generate_random_task_graph(num_nodes)
    task_graph = create_task_graph(node_weight, edge, edge_weight)

    pos = nx.shell_layout(task_graph)  # position
    nx.draw_networkx(task_graph, pos)
    plt.show()

def draw_task_graph_2(task_graph):
    pos = nx.shell_layout(task_graph)  # position
    nx.draw_networkx(task_graph, pos)
    plt.show()

def draw_task_graph_3(node_weight, edge, edge_weight):
    task_graph = create_task_graph(node_weight, edge, edge_weight)

    pos = nx.random_layout(task_graph)  # position
    nx.draw_networkx(task_graph, pos)
    plt.show()

if __name__ == "__main__":
    time_file = 'Resnet50/Resnet50_time.txt'
    bench_file = 'Resnet50/Resnet50_bench.txt'
    node_weight, s, s_weight = read_file.get_data(time_file, bench_file)
    draw_task_graph_3(node_weight, s, s_weight)