import random

def mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight):
    dic = {}
    # 随机分配任务到各个芯粒
    cuts = [[] for _ in lis_num_core]
    tasks = list(range(num_nodes))
    random.shuffle(tasks)
    for task in tasks:
        for i, cut in enumerate(cuts):
            if len(cut) < lis_num_core[i]:
                cut.append(task)
                break
    # 随机分配任务到核心
    for i, cut in enumerate(cuts):
        # 随机打乱任务顺序
        random.shuffle(cut)
        # 获取该芯粒的所有核心
        cores = [(core, i) for core in range(lis_num_core[i])]
        random.shuffle(cores)
        # 分配任务到核心
        for task, core in zip(cut, cores):
            dic[task+1] = {
                'chiplet': i+1,
                'core': core[0]+1
                # 'distance_to_d2d': random.uniform(1, 10)  # 随机生成一个距离值
            }
    return dic

# 辅助函数（保持与原代码一致）
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

# 示例用法（与原代码一致）
num_nodes = 5
s = {1: [2, 3], 2: [3], 3: [1, 2, 4], 4: [1, 3], 5: [4]}
s = modify_s(s)
s_weight = {(1, 2): 10, (1, 3): 20, (2, 3): 15, (3, 4): 25, (4, 5): 30, (1,4): 10}
s_weight = modify_s_weight(s_weight)
lis_num_core = [2, 3]
lis_num_d2d = [1, 2]
e = {
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
e = modify_e(e)
e_weight = {
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
e_weight = modify_e_weight(e_weight)
result = mapping(num_nodes, s, s_weight, lis_num_core, lis_num_d2d, e, e_weight)
print("-------------mapping result-------------")
print(result)