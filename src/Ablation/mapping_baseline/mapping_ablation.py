import math
from src import mapping, read_file, topology, random_mapping
from src.shortest_path import floyed_init
from src.dag import get_earlest_time

INF = 0x3f3f3f3f
num_nodes = 81
n = 9
num_core = 9
num_d2d = 1
N = n * (num_core + num_d2d)
num_add_edge_inter = (int(math.sqrt(n)) - 1) ** 2 + 6
num_add_edge_intra = (int(math.sqrt(num_core)) - 1) ** 2 + 6

def mapping_ablation(time_file, bench_file):
    # resnet50
    # ring
    # random mapping
    # 1. 任务图
    '''node_weight, s, s_weight = task.generate_random_task_graph(num_nodes)
    task_graph = task.create_task_graph(node_weight, s, s_weight)
    task.draw_task_graph(task_graph)'''

    # time_file = '../../Resnet50/Resnet50_time.txt'
    # bench_file = '../../Resnet50/Resnet50_bench.txt'
    # time_file = './src/Resnet50/Resnet50_time.txt'
    # bench_file = './src/Resnet50/Resnet50_bench.txt'

    node_weight, s, s_weight = read_file.get_data(time_file, bench_file)

    # 2. 拓扑图初始化
    e, e_weight = topology.generate_ring_ring_topology(n, num_core, num_d2d)

    # 3. 随机mapping
    '''map_dic = {}
    for i in range(1, 1 + 81):
        map_dic[i] = ((i - 1) // num_core) * (num_core + num_d2d) + (i - 1) % num_core + 1'''
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
    print(f'time consumption of random_mapping + ring-ring topology: {tmp_performance}')

    # resnet50
    # ring
    # proposed mapping
    # 1. 任务图
    node_weight, s, s_weight = read_file.get_data(time_file, bench_file)

    # 2. 拓扑图初始化
    e, e_weight = topology.generate_ring_ring_topology(n, num_core, num_d2d)

    # 3. 随机mapping
    '''map_dic = {}
    for i in range(1, 1 + 81):
        map_dic[i] = ((i - 1) // num_core) * (num_core + num_d2d) + (i - 1) % num_core + 1'''
    lis_num_core = [num_core for i in range(1 + n)]
    lis_num_d2d = [num_d2d for i in range(1 + n)]
    mapping_dict = mapping.mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight)
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
    print(f'time consumption of proposed mapping + ring-ring topology: {tmp_performance}')

    # resnet50
    # mesh
    # random mapping
    # 1. 任务图
    node_weight, s, s_weight = read_file.get_data(time_file, bench_file)

    # 2. 拓扑图初始化
    e, e_weight = topology.generate_mesh_mesh_topology(n, num_core, num_d2d)

    # 3. mapping
    '''map_dic = {}
    for i in range(1, 1 + 81):
        map_dic[i] = ((i - 1) // num_core) * (num_core + num_d2d) + (i - 1) % num_core + 1'''
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
    print(f'time consumption of random mapping + mesh-mesh topology: {tmp_performance}')

    # resnet50
    # mesh
    # proposed mapping
    # 1. 任务图
    node_weight, s, s_weight = read_file.get_data(time_file, bench_file)

    # 2. 拓扑图初始化
    e, e_weight = topology.generate_mesh_mesh_topology(n, num_core, num_d2d)

    # 3. mapping
    '''map_dic = {}
    for i in range(1, 1 + 81):
        map_dic[i] = ((i - 1) // num_core) * (num_core + num_d2d) + (i - 1) % num_core + 1'''
    lis_num_core = [num_core for i in range(1 + n)]
    lis_num_d2d = [num_d2d for i in range(1 + n)]
    mapping_dict = mapping.mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight)
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
    print(f'time consumption of proposed mapping + mesh-mesh topology: {tmp_performance}')

if __name__ == "__main__":
    time_file = '../../Resnet50/Resnet50_time.txt'
    bench_file = '../../Resnet50/Resnet50_bench.txt'
    mapping_ablation(time_file, bench_file)