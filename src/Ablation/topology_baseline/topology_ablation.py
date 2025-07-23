from src import read_file, topology, random_mapping
import math
from src.shortest_path import floyed_init
from src.dag import get_earlest_time
from src.greedy import greedy

INF = 0x3f3f3f3f
num_nodes = 81
n = 9
num_core = 9
num_d2d = 1
N = n * (num_core + num_d2d)
num_add_edge_inter = (int(math.sqrt(n)) - 1) ** 2 + 6
num_add_edge_intra = (int(math.sqrt(num_core)) - 1) ** 2 + 6

def topology_ablation(time_file, bench_file):
    # resnet50
    # ring
    # 1. 任务图
    # time_file = '../../Resnet50/Resnet50_time.txt'
    # bench_file = '../../Resnet50/Resnet50_bench.txt'
    node_weight, s, s_weight = read_file.get_data(time_file, bench_file)

    # 2. 拓扑图初始化
    e, e_weight = topology.generate_ring_ring_topology(n, num_core, num_d2d)

    # 3. mapping控制为一样
    lis_num_core = [num_core for i in range(1 + n)]
    lis_num_d2d = [num_d2d for i in range(1 + n)]
    mapping_dict = random_mapping.mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight)
    map_dic = {}
    for task, dic in mapping_dict.items():
        map_dic[task] = (dic['chiplet'] - 1) * (num_core + num_d2d) + dic['core']

    # 4. 求最短路径和最早达到时间
    dist = floyed_init(n, num_core, num_d2d, e, e_weight)
    t = [[0 for i in range(1 + num_nodes)] for j in range(1 + num_nodes)]
    for node, lis_dep in s.items():
        for pre_node in lis_dep:
            t[pre_node][node] = dist[map_dic[pre_node]][map_dic[node]] + s_weight[pre_node, node] + 1
    tmp_performance = get_earlest_time(num_nodes, t, node_weight, s)[0]
    print(f'time consumption of ring-ring topology: {tmp_performance}')

    # mesh
    # 1. 任务图

    # 2. 拓扑图初始化
    e, e_weight = topology.generate_mesh_mesh_topology(n, num_core, num_d2d)

    # 3. mapping控制为一样
    lis_num_core = [num_core for i in range(1 + n)]
    lis_num_d2d = [num_d2d for i in range(1 + n)]
    mapping_dict = random_mapping.mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight)
    map_dic = {}
    for task, dic in mapping_dict.items():
        map_dic[task] = (dic['chiplet'] - 1) * (num_core + num_d2d) + dic['core']

    # 4. 求最短路径和最早达到时间
    dist = floyed_init(n, num_core, num_d2d, e, e_weight)
    t = [[0 for i in range(1 + num_nodes)] for j in range(1 + num_nodes)]
    for node, lis_dep in s.items():
        for pre_node in lis_dep:
            t[pre_node][node] = dist[map_dic[pre_node]][map_dic[node]] + s_weight[pre_node, node] + 1
    tmp_performance = get_earlest_time(num_nodes, t, node_weight, s)[0]
    print(f'time consumption of mesh-mesh topology: {tmp_performance}')

    # proposed
    # 1. 任务图

    # 2. 拓扑图初始化
    e, e_weight = topology.generate_mesh_mesh_topology(n, num_core, num_d2d)

    # 3. mapping控制为一样
    lis_num_core = [num_core for i in range(1 + n)]
    lis_num_d2d = [num_d2d for i in range(1 + n)]
    mapping_dict = random_mapping.mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight)
    map_dic = {}
    for task, dic in mapping_dict.items():
        map_dic[task] = (dic['chiplet'] - 1) * (num_core + num_d2d) + dic['core']

    # 4. 求最短路径和最早达到时间
    dist = floyed_init(n, num_core, num_d2d, e, e_weight)
    t = [[0 for i in range(1 + num_nodes)] for j in range(1 + num_nodes)]
    for node, lis_dep in s.items():
        for pre_node in lis_dep:
            t[pre_node][node] = dist[map_dic[pre_node]][map_dic[node]] + s_weight[pre_node, node] + 1
    tmp_performance = get_earlest_time(num_nodes, t, node_weight, s)[0]
    # print(f'mesh-mesh的performance：{tmp_performance}')
    tmp_solution = [[INF for i in range(1 + N)] for j in range(1 + N)]
    for i in range(1 + N):
        tmp_solution[i][i] = 0
    for node, lis_dep in e.items():
        for next_node in lis_dep:
            first = (node[0] - 1) * (num_core + num_d2d) + node[1]
            second = (next_node[0] - 1) * (num_core + num_d2d) + next_node[1]
            tmp_solution[first][second] = tmp_solution[second][first] = e_weight[node, next_node]

    performance, solution, dp = greedy(num_nodes, node_weight, s, s_weight, n, num_core, num_d2d, map_dic, num_add_edge_intra, num_add_edge_inter, tmp_performance, tmp_solution)
    print(f'final optimized time consumption: {performance}')

if __name__ == "__main__":
    topology_ablation('../../Resnet50/Resnet50_time.txt', '../../Resnet50/Resnet50_bench.txt')