'''
input:包括task-graph和topology

task-graph:
num_nodes: int : 任务节点数量
node_weight: list : num_weight[i]表示task i的执行时间
s: dict : s[i] = [j1,...,jn]表示task i的边相连的边（邻接表）
s_weight : dict : s_weight[(j1, j2)] = weight 表示task j1 与 j2 的边的通信量为weight

topology:
n:
num_core:
num_d2d:
e: dict : e[i,j] = [(i1,j1),...,()] 边的连接
e_weight : dict : e[(i1,j1),(i2,j2)] = weight

output: 映射字典
return dic # dic[i] = (j,k)

def mapping(num_nodes, node_weight, s, s_weight, num_core, num_d2d, e, e_weight):
    dic = {}
    return dic
'''

import heapq


def dijkstra(graph, start):
    min_heap = [(0, start)]
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    while min_heap:
        current_distance, current_vertex = heapq.heappop(min_heap)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(min_heap, (distance, neighbor))
    return distances


def calculate_core_to_d2d_distances(lis_num_core, lis_num_d2d, e, e_weight):
    core_to_d2d_distances = {}
    for i in range(len(lis_num_core)):
        graph = {}
        for core in range(lis_num_core[i]):
            graph[(core, i)] = {}
        for d2d_index in range(lis_num_d2d[i]):
            graph[(lis_num_core[i] + d2d_index, i)] = {}

        for (u, v), weight in e_weight.items():
            if u[1] == i and v[1] == i:
                u_key = (u[0], i)
                v_key = (v[0], i)
                if u_key not in graph:
                    graph[u_key] = {}
                if v_key not in graph:
                    graph[v_key] = {}
                graph[u_key][v_key] = weight
                graph[v_key][u_key] = weight

        for core in range(lis_num_core[i]):
            distances = dijkstra(graph, (core, i))
            # print(f"Distances from core {core} in chiplet {i}: {distances}")
            min_distance = float('inf')
            for d2d_index in range(lis_num_d2d[i]):
                d2d_core = (lis_num_core[i] + d2d_index, i)
                if d2d_core in distances:
                    distance = distances[d2d_core]
                    if distance < min_distance:
                        min_distance = distance
            core_to_d2d_distances[(core, i)] = min_distance
            # print(f"Minimum distance from core {core} to D2D in chiplet {i}: {min_distance}")
    return core_to_d2d_distances


def calculate_comm_volume(cuts, s_weight):
    total_comm_volume = 0
    for p in range(len(cuts)):
        for q in range(p + 1, len(cuts)):
            for v_r in cuts[p]:
                for v_s in cuts[q]:
                    total_comm_volume += s_weight.get((v_r, v_s), 0)
    return total_comm_volume


def swap_and_evaluate(cuts, s_weight):
    original_comm_volume = calculate_comm_volume(cuts, s_weight)
    while True:
        swapped = 0
        for p in range(len(cuts)):
            if swapped:
                break
            for q in range(p + 1, len(cuts)):
                if swapped:
                    break
                for vj_index in range(len(cuts[p])):
                    if swapped:
                        break
                    for vk_index in range(len(cuts[q])):
                        vj = cuts[p][vj_index]
                        vk = cuts[q][vk_index]

                        # 交换尝试
                        cuts[p][vj_index] = vk
                        cuts[q][vk_index] = vj

                        new_comm_volume = calculate_comm_volume(cuts, s_weight)
                        if new_comm_volume < original_comm_volume:
                            original_comm_volume = new_comm_volume  # 更新最小通信量并保持交换
                            swapped = 1
                            break

                        else:
                            # 撤销交换
                            cuts[p][vj_index] = vj
                            cuts[q][vk_index] = vk
        if swapped == 0:  # 不再有交换了，就退出
            break
        return cuts


def mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight):
    dic = {}
    # Step 1: Task Graph Partitioning
    cuts = [[] for _ in lis_num_core]
    total_comm_volume = {}
    for i in range(num_nodes):
        total_comm_volume[i] = sum(s_weight.get((i, j), 0) for j in s.get(i, []))

    sorted_tasks = sorted(total_comm_volume.items(), key=lambda x: x[1], reverse=True)
    # print("sorted tasks: ")
    # print(sorted_tasks)

    for i, (task, _) in enumerate(sorted_tasks):
        for j, cut in enumerate(cuts):
            if len(cut) < lis_num_core[j]:
                cut.append(task)
                break

    # Step 2: Node Swapping Adjustment
    cuts = swap_and_evaluate(cuts, s_weight)

    # 第3步：芯粒内部映射
    core_to_d2d_distances = calculate_core_to_d2d_distances(lis_num_core, lis_num_d2d, e, e_weight)
    for i, cut in enumerate(cuts):
        F_values = {}
        for task in cut:
            inter_cut_comm = sum(s_weight.get((task, v), 0) for k, cut_k in enumerate(cuts) if k != i for v in cut_k)
            # print(inter_cut_comm)
            inta_cut_comm = sum(s_weight.get((task, v), 0) for v in cut if v != task)
            # print(inta_cut_comm)
            F_values[task] = inter_cut_comm - inta_cut_comm

        # print("F_values:")
        # print(F_values)
        sorted_tasks_by_F = sorted(F_values, key=F_values.get, reverse=True)

        sorted_cores_by_d2d = sorted([k for k in core_to_d2d_distances if k[1] == i],
                                     key=lambda x: core_to_d2d_distances[x])
        # print("sorted_cores_by_d2d:")
        # print(sorted_cores_by_d2d)
        # print("sorted_tasks_by_F :")
        # print(sorted_tasks_by_F)

        for task, core in zip(sorted_tasks_by_F, sorted_cores_by_d2d):
            dic[task+1] = {'chiplet': i+1, 'core': core[0]+1, 'distance_to_d2d': core_to_d2d_distances[core]}
            # dic[task] = {'chiplet': i, 'core': core[0]}
    return dic


# 示例用法：
# 核的索引是 (核编号, 芯粒编号)，D2D 接口的索引是 (核编号 + 总核数, 芯粒编号)。

def modify_s(s):
    new_s = {}
    for key, values in s.items():
        new_s[key - 1] = [value - 1 for value in values]
    return new_s

def modify_s_weight(s_weight):
    new_s_weight = {}
    for key, value in s_weight.items():
        new_key = tuple(x - 1 for x in key)
        new_s_weight[new_key] = value
    return new_s_weight

def modify_e(e):
    new_e = {}
    for key, values in e.items():
        new_key = tuple(x - 1 for x in key)
        new_values = [tuple(v - 1 for v in value) for value in values]
        new_e[new_key] = new_values
    return new_e

def modify_e_weight(e_weight):
    new_e_weight = {}
    for (key1, key2), weight in e_weight.items():
        new_key1 = tuple(x - 1 for x in key1)
        new_key2 = tuple(x - 1 for x in key2)
        new_e_weight[(new_key1, new_key2)] = weight
    return new_e_weight

num_nodes = 5
s = {1: [2, 3], 2: [3], 3: [1, 2, 4], 4: [1, 3], 5: [4]}  # 任务图的节点连接关系，编号从1开始到num_nodes，采用邻接表的形式
s=modify_s(s)
# print(s)

s_weight = {(1, 2): 10, (1, 3): 20, (2, 3): 15, (3, 4): 25, (4, 5): 30, (1,4): 10}  # 任务图边的权重
s_weight=modify_s_weight(s_weight)
# print(s_weight)

lis_num_core = [2, 3]   # 每个芯粒核的数目，2个芯粒
lis_num_d2d = [1, 2]    # 每个芯粒D2D接口数目

e = {  # 拓扑图中边的连接，邻接表
(2, 2): [(1, 1), (2, 3), (1, 2), (3, 2)],
(1, 1): [(2, 2), (3, 1)],
(2, 3): [(2, 2), (3, 3)],
(3, 3): [(2, 3)],
(3, 1): [(1, 1), (2, 1)],
(1, 2): [(4, 2), (5, 2), (2, 2)],
(4, 2): [(1, 2)],
(5, 2): [(1, 2)],
(2, 1): [(3, 1)],
(3, 2): [(2, 2)]
}
e=modify_e(e)
# print(e)

e_weight = {   # 拓扑图中边的权重
    ((2, 2), (1, 1)): 5,
    ((2, 2), (2, 3)): 10,
    ((2, 3), (3, 3)): 15,
    ((3, 1), (1, 1)): 9,
    ((1, 2), (4, 2)): 9,
    ((1, 2), (5, 2)): 9,
    ((2, 1), (3, 1)): 9,
    ((1, 2), (2, 2)): 100,
    ((2, 2), (3, 2)): 77
}
e_weight=modify_e_weight(e_weight)
# print(e_weight)

result = mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight)
# print("-------------mapping result-------------")
# print(result)
