def get_data(time_file, bench_file):
    node_weight = [0]
    s = {}
    s_weight = {}

    f = open(time_file)
    lines = f.readlines()
    n = len(lines)
    for i in range(n):
        node_weight.append(int(lines[i]))
    f.close()

    for i in range(1, 1 + n):
        s[i] = []

    f = open(bench_file)
    lines = f.readlines()
    bench_length = len(lines)

    for i in range(bench_length):
        data = lines[i].split()
        min_val = min(int(data[1]), int(data[0]))
        max_val = max(int(data[1]), int(data[0]))
        s[max_val].append(min_val)
        s_weight[min_val, max_val] = float(data[2])
    f.close()

    return node_weight, s, s_weight

if __name__ == "__main__":
    time_file = './Resnet50/Resnet50_time.txt'
    bench_file = './Resnet50/Resnet50_bench.txt'
    node_weight, s, s_weight = get_data(time_file, bench_file)
    total = 0
    for k, v in s_weight.items():
        total += v
    print(f"total:{total}")
    print(node_weight)
    print(s)
    print(s_weight)
