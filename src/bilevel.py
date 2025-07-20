from src import read_file
import math
from src import topology
from src.mapping import mapping
from src.shortest_path import floyed_init
from src.dag import get_earlest_time
from src.greedy import greedy

INF = 0x3f3f3f3f
num_nodes = 81
n = 9
num_core = 9
num_d2d = 1
N = n * (num_core + num_d2d)
num_add_edge_inter = (int(math.sqrt(n)) - 1) ** 2 + 1
num_add_edge_intra = (int(math.sqrt(num_core)) - 1) ** 2 + 1
'''num_add_edge_inter = 2
num_add_edge_intra = 2'''

def map_no_d2d(x):
    if x % 10 == 0:
        group_index = (x - 1) // 10 + 1
        return (group_index - 1) * 9 + 5
    else:
        group_index = (x - 1) // 10 + 1
        within_index = x % 10
        return (group_index - 1) * 9 + within_index


def bilevel(time_file, bench_file, trace_file, topo_file, show_file):
    # 1. 任务图

    '''node_weight, s, s_weight = task.generate_random_task_graph(num_nodes)
    task_graph = task.create_task_graph(node_weight, s, s_weight)
    task.draw_task_graph(task_graph)'''

    # time_file = './CNN/CNN_time.txt'
    # time_file = './MLP/MLP_time.txt'
    # time_file = './Resnet50/Resnet50_time.txt'
    # time_file = './VGG/VGG_time.txt'

    # bench_file = './CNN/CNN_bench.txt'
    # bench_file = './MLP/MLP_bench.txt'
    # bench_file = './Resnet50/Resnet50_bench.txt'
    # bench_file = './VGG/VGG_bench.txt'


    node_weight, s, s_weight = read_file.get_data(time_file, bench_file)
    # task.draw_task_graph_3(node_weight, s, s_weight)

    # 2. 拓扑图初始化
    e, e_weight = topology.generate_ring_ring_topology(n, num_core, num_d2d)

    # 3. mapping
    lis_num_core = [num_core for i in range(1 + n)]
    lis_num_d2d = [num_d2d for i in range(1 + n)]
    mapping_dict = mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight)
    map_dic = {}
    for task, dic in mapping_dict.items():
        map_dic[task] = (dic['chiplet'] - 1) * (num_core + num_d2d) + dic['core']
    '''
    print("mapping:")
    for k, v in map_dic.items():
        print(f'{k}:{v}')
    '''

    # 4. 求最短路径和最早达到时间
    dist = floyed_init(n, num_core, num_d2d, e, e_weight)
    t = [[0 for i in range(1 + num_nodes)] for j in range(1 + num_nodes)]
    for node, lis_dep in s.items():
        for pre_node in lis_dep:
            t[pre_node][node] = dist[map_dic[pre_node]][map_dic[node]] + s_weight[pre_node, node] + 1
    tmp_performance = get_earlest_time(num_nodes, t, node_weight, s)[0]
    tmp_solution = [[INF for i in range(1 + N)] for j in range(1 + N)]
    for i in range(1 + N):
        tmp_solution[i][i] = 0
    for node, lis_dep in e.items():
        for next_node in lis_dep:
            first = (node[0] - 1) * (num_core + num_d2d) + node[1]
            second = (next_node[0] - 1) * (num_core + num_d2d) + next_node[1]
            tmp_solution[first][second] = tmp_solution[second][first] = e_weight[node, next_node]
    print(f's-s的performance：{tmp_performance}')

    # 5. 贪心搜索
    performance, solution, dp = greedy(num_nodes, node_weight, s, s_weight, n, num_core, num_d2d, map_dic, num_add_edge_intra, num_add_edge_inter, tmp_performance, tmp_solution)

    # trace_file = './CNN/CNN_trace.txt'
    # trace_file = './MLP/MLP_trace.txt'
    # trace_file = './Resnet50/Resnet50_trace.txt'
    # trace_file = './VGG/VGG_trace.txt'

    with open(trace_file, 'w') as f:
        for node, lis_next in s.items():
            for pre_node in lis_next:
                f.write(f'{int(dp[pre_node])} {map_dic[pre_node]} {map_dic[node]} {int(s_weight[pre_node, node])}\n')
    f.close()

    sorted_list = []
    with open(trace_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            sorted_list.append([])
            lis = line.split()
            for element in lis:
                sorted_list[i].append(int(element))
    f.close()

    sorted_list.sort(key=lambda lis: lis[0])

    with open(trace_file, 'w') as f:
        for lis in sorted_list:
            f.write(f'{lis[0]} {lis[1]} {lis[2]} {lis[3]}\n')
    f.close()



    print(f'搜索的performance：{performance}')
    print('搜索的topology')
    res = []
    for i in range(1 + N):
        res.append([])
    for i in range(1, 1 + N):
        for j in range(1, i):
            if solution[j][i] != 0 and solution[j][i] != INF:
                res[i].append(j)

    # topo_file = './CNN/CNN_gv.txt'
    # topo_file = './MLP/MLP_gv.txt'
    # topo_file = './Resnet50/Resnet50_gv.txt'
    # topo_file = './VGG/VGG_gv.txt'
    with open(topo_file, 'w') as f:
        for i in range(1, 1 + N):
            print(f'{i}:{res[i]}')
            for j in res[i]:
                f.write(f'{j}--{i}\n')

    # show_file = './CNN/CNN_show.txt'
    # show_file = './MLP/MLP_show.txt'
    # show_file = './Resnet50/Resnet50_show.txt'
    # show_file = './VGG/VGG_show.txt'
    with open(show_file, 'w') as f:
        for i in range(1, 1 + N):
            for j in res[i]:
                ii = map_no_d2d(i)
                jj = map_no_d2d(j)
                if ii != jj:
                    f.write(f'{jj}--{ii}\n')

    '''
    e, e_weight = topology.generate_mesh_mesh_topology(n, num_core, num_d2d)
    dist = floyed_init(n, num_core, num_d2d, e, e_weight)
    t = [[0 for i in range(1 + num_nodes)] for j in range(1 + num_nodes)]
    for node, lis_dep in s.items():
        for pre_node in lis_dep:
            t[pre_node][node] = dist[map_dic[pre_node]][map_dic[node]] + s_weight[pre_node, node] + 1
    mesh_performance = get_earlest_time(num_nodes, t, node_weight, s)
    print(f'mesh-mesh的performance：{performance}')
    '''

if __name__ == '__main__':
    time_file = './CNN/CNN_time.txt'
    bench_file = './CNN/CNN_bench.txt'
    trace_file = './CNN/CNN_trace.txt'
    topo_file = './CNN/CNN_gv.txt'
    show_file = './CNN/CNN_show.txt'
    bilevel(time_file, bench_file, trace_file, topo_file, show_file)