import queue
import copy

def get_earlest_time(n, t, node_weight, s):
    q = queue.Queue()
    dp = [0 for i in range(1 + n)]

    indegree = [0 for i in range(1 + n)]
    outdegree = [0 for i in range(1 + n)]
    lis_pre = []
    lis_next = []
    for i in range(1 + n):
        lis_pre.append([])
        lis_next.append([])

    for node, lis_node in s.items():
        for pre_node in lis_node:
            indegree[node] += 1
            outdegree[pre_node] += 1
            lis_pre[node].append(pre_node)
            lis_next[pre_node].append(node)
    '''for i in range(1, 1 + n):
        for j in range(1, 1 + n):
            if t[i][j]:
                indegree[j] += 1
                outdegree[i] += 1
                lis_pre[j].append(i)
                lis_next[i].append(j)'''

    for i in range(1, 1 + n):
        if not indegree[i]:
            lis_pre[i].append(0)
            lis_next[0].append(i)
            q.put(i)

    while not q.empty():
        current = q.get()

        for pre in lis_pre[current]:
            dp[current] = max(dp[current], dp[pre] + t[pre][current] + node_weight[current])
            # print(f'dp[{current}]:{dp[current]}')

        for next in lis_next[current]:
            indegree[next] -= 1
            if indegree[next] == 0:
                q.put(next)

    max_val = 0
    for i in range(1, 1 + n):
        max_val = max(max_val, dp[i])

    return max_val, copy.deepcopy(dp)