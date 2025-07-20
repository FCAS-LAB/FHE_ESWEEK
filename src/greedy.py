import copy
from src.shortest_path import floyed
from src.dag import get_earlest_time

INF = 0x3f3f3f3f
intra_weight = 1
inter_weight = 21.3

def get_core_start_end(num_core, num_d2d, x):
    N = num_core + num_d2d
    start = (x - 1) * N + 1
    end = (x - 1) * N + num_core
    return start, end

def get_d2d_list(n, num_core, num_d2d):
    N = num_core + num_d2d
    lis = []
    for i in range(1, 1 + n):
        for j in range(1 + num_core , 1 + N):
            lis.append((i - 1) * N + j)
    return lis

def is_same_chiplet(x, y, num_core, num_d2d):
    N = num_core + num_d2d
    return (x - 1) // N == (y - 1) // N

def greedy(num_nodes, node_weight, s, s_weight, n, num_core, num_d2d, map_dic, num_add_edge_intra, num_add_edge_inter, tmp_performance, tmp_solution):
    N = n * (num_core + num_d2d)
    performance = tmp_performance
    solution = copy.deepcopy(tmp_solution)
    dp = []

    d2d_lis = get_d2d_list(n, num_core, num_d2d)
    l = len(d2d_lis)

    for i in range(1, 1 + n):
        for j in range(num_add_edge_intra):
            start, end = get_core_start_end(num_core, num_d2d, i)
            current_performance = performance
            current_solution = copy.deepcopy(solution)

            # 尝试
            for x in range(start, end + 1):
                for y in range(x + 1, end + 1):
                    if solution[x][y] != INF:
                        continue
                    adj_matrix = copy.deepcopy(solution)
                    adj_matrix[x][y] = adj_matrix[y][x] = intra_weight
                    dist = floyed(N, adj_matrix)
                    t = [[0 for a in range(1 + num_nodes)] for b in range(1 + num_nodes)]
                    for node, lis_dep in s.items():
                        for pre_node in lis_dep:
                            t[pre_node][node] = dist[map_dic[pre_node]][map_dic[node]] + s_weight[pre_node, node] + 1
                        time, dp = get_earlest_time(num_nodes, t, node_weight, s)

                    if time <= current_performance:
                        current_performance = time
                        current_solution = copy.deepcopy(adj_matrix)

            # 更新
            performance = current_performance
            solution = copy.deepcopy(current_solution)
            '''
            print("------------------------------------------------------------------------------")
            print(performance)
            print(solution)
            '''
            print(f'{(i - 1) * num_add_edge_intra + j} insert:{performance}')

    for i in range(num_add_edge_inter):
        current_performance = performance
        current_solution = copy.deepcopy(solution)
        for x in range(l):
            for y in range(x + 1, l):
                if not is_same_chiplet(d2d_lis[x], d2d_lis[y],num_core, num_d2d):
                    if solution[d2d_lis[x]][d2d_lis[y]] != INF:
                        continue
                    adj_matrix = copy.deepcopy(solution)
                    adj_matrix[d2d_lis[x]][d2d_lis[y]] = adj_matrix[d2d_lis[y]][d2d_lis[x]] = inter_weight
                    dist = floyed(N, adj_matrix)
                    t = [[0 for a in range(1 + num_nodes)] for b in range(1 + num_nodes)]
                    for node, lis_dep in s.items():
                        for pre_node in lis_dep:
                            t[pre_node][node] = dist[map_dic[pre_node]][map_dic[node]] + s_weight[pre_node, node] / 16 + 1
                    time = get_earlest_time(num_nodes, t, node_weight, s)[0]
                    if time <= current_performance:
                        current_performance = time
                        current_solution = copy.deepcopy(adj_matrix)

        performance = current_performance
        solution = current_solution.copy()
        '''print("------------------------------------------------------------------------------")
        print(performance)
        print(solution)'''
        print(f'{num_add_edge_intra * n + i} insert:{performance}')

    return performance, solution, copy.deepcopy(dp)
