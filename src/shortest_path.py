import copy

INF = 0x3f3f3f3f

def init(n, num_core, num_d2d, e, e_weight):
    N = n * (num_core + num_d2d)
    dist = [[INF for i in range(1 + N)] for j in range(1 + N)]
    for i in range(1, 1 + N):
        dist[i][i] = 0

    for node, lis_dep in e.items():
        for next_node in lis_dep:
            first = (node[0] - 1) * (num_core + num_d2d) + node[1]
            second = (next_node[0] - 1) * (num_core + num_d2d) + next_node[1]
            dist[first][second] = dist[second][first] = e_weight[node, next_node]

    return dist

def floyed_init(n, num_core, num_d2d, e, e_weight):
    N = n * (num_core + num_d2d)
    dist = init(n, num_core, num_d2d, e, e_weight)

    for k in range(1, 1 + N):
        for i in range(1, 1 + N):
            for j in range(1, 1 + N):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

def floyed(n, adj_matrix):
    dist = copy.deepcopy(adj_matrix)
    for k in range(1, 1 + n):
        for i in range(1, 1 + n):
            for j in range(1, 1 + n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    return dist
