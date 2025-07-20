import math
import random

intra_weight = 1
inter_weight = 21.3

def generate_mesh_mesh_topology(n, num_core, num_d2d):
    '''
    1 -- 2
    |    |
    3 -- 4     5(d2d)
    '''
    sqrt_n = int(math.sqrt(n))
    sqrt_num_core = int(math.sqrt(num_core))
    e = {}
    for i in range(1, 1 + n):
        for j in range(num_core + num_d2d + 1):
            e[i, j] = []
    e_weight = {}

    for i in range(1, 1 + n):  # 片内互连
        # sram(j = 0) 与 core互连
        random_core_idx_sram = int(random.random() * num_core) + 1
        e[i, 0].append((i, random_core_idx_sram))
        e[i, random_core_idx_sram].append((i, 0))
        e_weight[(i, 0), (i, random_core_idx_sram)] = e_weight[(i, random_core_idx_sram), (i, 0)] = intra_weight

        # core互连
        for r in range(sqrt_num_core):
            for c in range(sqrt_num_core):
                j = r * sqrt_num_core + c + 1
                if r != 0:
                    e[i, j].append((i, j - sqrt_num_core))
                    e_weight[(i, j), (i, j - sqrt_num_core)] = e_weight[(i, j - sqrt_num_core), (i, j)] = intra_weight
                if c != 0:
                    e[i, j].append((i, j - 1))
                    e_weight[(i, j), (i, j - 1)] = e_weight[(i, j - 1), (i, j)] = intra_weight
                if c != sqrt_num_core - 1:
                    e[i, j].append((i, j + 1))
                    e_weight[(i, j), (i, j + 1)] = e_weight[(i, j + 1), (i, j)] = intra_weight
                if r != sqrt_num_core - 1:
                    e[i, j].append((i, j + sqrt_num_core))
                    e_weight[(i, j), (i, j + sqrt_num_core)] = e_weight[(i, j + sqrt_num_core), (i, j)] = intra_weight

        # core 与 d2d 互连
        for j in range(1 + num_core, 1 + num_core + num_d2d):
            random_core_idx_d2d = int(random.random() * num_core) + 1
            e[i, j].append((i, random_core_idx_d2d))
            e[i, random_core_idx_d2d].append((i, j))
            e_weight[(i, j), (i, random_core_idx_d2d)] = e_weight[(i, random_core_idx_d2d), (i, j)] = intra_weight

    # d2d接口相连
    if num_d2d == 1:
        j = 1 + num_core
        for r in range(sqrt_n):
            for c in range(sqrt_n):
                i = r * sqrt_n + c + 1
                if r != 0:
                    e[i, j].append((i - sqrt_n, j))
                    e_weight[(i, j), (i - sqrt_n, j)] = inter_weight
                if c != 0:
                    e[i, j].append((i - 1, j))
                    e_weight[(i, j), (i - 1, j)] = inter_weight
                if c != sqrt_n - 1:
                    e[i, j].append((i + 1, j))
                    e_weight[(i, j), (i + 1, j)] = inter_weight
                if r != sqrt_n - 1:
                    e[i, j].append((i + sqrt_n, j))
                    e_weight[(i, j), (i + sqrt_n, j)] = inter_weight
    elif num_d2d == 2:
        j1 = 1 + num_core  # 上下
        j2 = 2 + num_core  # 左右
        for r in range(sqrt_n):
            for c in range(sqrt_n):
                i = r * sqrt_n + c + 1
                if r != 0:
                    e[i, j1].append((i - sqrt_n, j1))
                    e_weight[(i, j1), (i - sqrt_n, j1)] = inter_weight
                if c != 0:
                    e[i, j2].append((i - 1, j2))
                    e_weight[(i, j2), (i - 1, j2)] = inter_weight
                if c != sqrt_n - 1:
                    e[i, j2].append((i + 1, j2))
                    e_weight[(i, j2), (i + 1, j2)] = inter_weight
                if r != sqrt_n - 1:
                    e[i, j1].append((i + sqrt_n, j1))
                    e_weight[(i, j1), (i + sqrt_n, j1)] = inter_weight
    elif num_d2d == 4:
        j1 = 1 + num_core  # 上
        j2 = 2 + num_core  # 左
        j3 = 3 + num_core  # 右
        j4 = 4 + num_core  # 下
        for r in range(sqrt_n):
            for c in range(sqrt_n):
                i = r * sqrt_n + c + 1
                if r != 0:
                    e[i, j1].append((i - sqrt_n, j4))
                    e_weight[(i, j1), (i - sqrt_n, j4)] = inter_weight
                if c != 0:
                    e[i, j2].append((i - 1, j3))
                    e_weight[(i, j2), (i - 1, j3)] = inter_weight
                if c != sqrt_n - 1:
                    e[i, j3].append((i + 1, j2))
                    e_weight[(i, j3), (i + 1, j2)] = inter_weight
                if r != sqrt_n - 1:
                    e[i, j4].append((i + sqrt_n, j1))
                    e_weight[(i, j4), (i + sqrt_n, j1)] = inter_weight

    return e, e_weight

def generate_ring_ring_topology(n, num_core, num_d2d):
    sqrt_n = int(math.sqrt(n))
    sqrt_num_core = int(math.sqrt(num_core))

    e = {}
    e_weight = {}

    lis_right_n = [i for i in range(1, 1 + n) if i % sqrt_n != 0]
    lis_down_n = [i for i in range(1, 1 + n - sqrt_n) if ((i - 1) // sqrt_n) % 2 == (i % sqrt_n)]
    lis_right_num_core = [i for i in range(1, 1 + num_core) if i % sqrt_num_core != 0]
    lis_down_num_core = [i for i in range(1, 1 + num_core - sqrt_n) if ((i - 1) // sqrt_num_core) % 2 == (i % sqrt_num_core)]

    '''
    print(lis_right_n)
    print(lis_down_n)
    print(lis_right_num_core)
    print(lis_down_num_core)
    '''

    for i in range(1, 1 + n):
        for j in range(1, 1 + num_core + num_d2d):
            e[i, j] = []

    # 片内
    for i in range(1, 1 + n):
        for j in range(1, 1 + num_core):
            if j in lis_right_num_core:
                e[i, j].append((i, j + 1))
                e[i, j + 1].append((i, j))
                e_weight[(i, j), (i, j + 1)] = e_weight[(i, j + 1), (i, j)] = intra_weight
            if j in lis_down_num_core:
                e[i, j].append((i, j + sqrt_num_core))
                e[i, j + sqrt_num_core].append((i, j))
                e_weight[(i, j), (i, j + sqrt_num_core)] = e_weight[(i, j + sqrt_num_core), (i, j)] = intra_weight

    # 片间
    mid_idx = (num_core + 1) // (num_d2d + 1)
    lis_mid_idx = [mid_idx * i for i in range(1, 1 + num_d2d)]
    lis_d2d_idx = [i for i in range(1 + num_core, 1 + num_core + num_d2d)]

    for i in range(1, 1 + n):
        for j in range(num_d2d):
            e[i, lis_mid_idx[j]].append((i, lis_d2d_idx[j]))
            e[i, lis_d2d_idx[j]].append((i, lis_mid_idx[j]))
            e_weight[(i, lis_mid_idx[j]), (i, lis_d2d_idx[j])] = e_weight[(i, lis_d2d_idx[j]), (i, lis_mid_idx[j])] = intra_weight

            if i in lis_right_n:
                e[i, lis_d2d_idx[j]].append((i + 1, lis_d2d_idx[j]))
                e[i + 1, lis_d2d_idx[j]].append((i, lis_d2d_idx[j]))
                e_weight[(i, lis_d2d_idx[j]), (i + 1, lis_d2d_idx[j])] = e_weight[(i + 1, lis_d2d_idx[j]), (i, lis_d2d_idx[j])] = inter_weight
            if i in lis_down_n:
                e[i, lis_d2d_idx[j]].append((i + sqrt_n, lis_d2d_idx[j]))
                e[i + sqrt_n, lis_d2d_idx[j]].append((i, lis_d2d_idx[j]))
                e_weight[(i, lis_d2d_idx[j]), (i + sqrt_n, lis_d2d_idx[j])] = e_weight[(i + sqrt_n, lis_d2d_idx[j]), (i, lis_d2d_idx[j])] = inter_weight

    return e, e_weight


'''
if __name__ == "__main__":
    e, e_weight = generate_ring_ring_topology(16, 16, 4)
    print(e)
    print(e_weight)
'''