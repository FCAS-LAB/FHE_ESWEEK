import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 创建一个图
G = nx.Graph()

# 添加81个节点
nodes = list(range(1, 82))  # 节点 1 到 81
G.add_nodes_from(nodes)

# 定义9x9的矩阵排列位置
positions = {}

matrix_size = 9  # 9x9的矩阵
group_size = 3  # 每组3x3

# 为81个节点分配3x3网格的位置
for i in range(81):
    # 计算当前节点所在的组
    group_index = i // 9
    within_group_index = i % 9

    # 每组内的行列
    row_within_group = within_group_index // group_size
    col_within_group = within_group_index % group_size

    # 每个组内的节点按3x3排列，外部的行列位置
    row = group_index // 3 * group_size + row_within_group
    col = group_index % 3 * group_size + col_within_group

    positions[nodes[i]] = (col, -row)  # 使用负号使得图从上到下排列

'''
# 添加组内的环形连接（不闭合，最后一个节点不与第一个节点连接）
for i in range(81):
    group_index = i // 9
    within_group_index = i % 9

    # 每组内的行列
    row_within_group = within_group_index // group_size
    col_within_group = within_group_index % group_size

    # 每组内部的节点按指定顺序连接：1--2--3--6--5--4--7--8--9（不闭合）
    connection_order = [0, 1, 2, 5, 4, 3, 6, 7, 8]  # 按照 1--2--3--6--5--4--7--8--9 顺序
    for j in range(len(connection_order) - 1):
        G.add_edge(nodes[i - within_group_index + connection_order[j]], nodes[i - within_group_index + connection_order[j + 1]])

# 获取每个小组的正中心节点
center_nodes = []

# 获取每个小组的正中心节点（即第5个节点）
for i in range(9):
    group_start = i * 9
    center_node = group_start + 4  # 每个小组的正中心节点是第5个节点
    center_nodes.append(nodes[center_node])

# 将片间正中心节点按顺序连接（不闭合，9个正中心节点相连）
# 片间连接顺序：5--14--23--50--41--32--59--68--77
inter_group_connections = [5, 14, 23, 50, 41, 32, 59, 68, 77]
for i in range(len(inter_group_connections) - 1):  # 不闭合
    current_center = inter_group_connections[i]
    next_center = inter_group_connections[i + 1]
    G.add_edge(current_center, next_center)
'''
inter_group_connections = [5, 14, 23, 50, 41, 32, 59, 68, 77]
inter_list = []
show_file = '../Resnet50/Resnet50_show.txt'
f = open(show_file)
lines = f.readlines()
for line in lines:
    a, b = line.split('--')
    a = int(a)
    b = int(b)
    if a in inter_group_connections and b in inter_group_connections:
        inter_list.append((a, b))
    G.add_edge(a, b)


# 绘制图形，显示节点和边
plt.figure(figsize=(12, 12))
ax = plt.gca()

# 绘制节点（只显示节点位置）
nx.draw_networkx_nodes(G, pos=positions, node_size=1400, node_color='lightblue', ax=ax)

# 绘制节点标签
nx.draw_networkx_labels(G, pos=positions, font_size=15, font_weight='bold', ax=ax)

# 绘制片内的环形连接（蓝色）
nx.draw_networkx_edges(G, pos=positions, width=1.5, alpha=0.6, edge_color='blue', ax=ax)

# 绘制片间正中心节点的连接（红色）
nx.draw_networkx_edges(G, pos=positions, edgelist=inter_list, width=2.5, alpha=0.8, edge_color='red', ax=ax)
print(inter_list)

# 绘制每组的边框（每组9个节点为一组）
for i in range(9):
    # 计算每组的坐标范围
    group_start_x = (i % 3) * 3  # 每组横坐标
    group_start_y = (i // 3) * 3  # 每组纵坐标

    # 画矩形框
    ax.add_patch(Rectangle((group_start_x - 0.5, -group_start_y - 2.5), 3, 3, linewidth=3, edgecolor='black', facecolor='none'))

# 关闭坐标轴
plt.axis('off')
plt.show()
