# 全同态加密任务图映射与片间/片内互连网络拓扑生成框架

本代码库提供了一个基于 Python 的框架，用于在多芯粒（multi-chiplet）系统中对全同态加密（Fully Homomorphic Encryption, FHE) 应用进行设计空间探索 (DSE)。它包含双层规划算法、映射和拓扑生成消融实验以及相关的实验脚本。

## 核心功能

- **双层规划算法**: 联合优化映射和拓扑。
- **基线消融实验**: 对映射和拓扑进行单独的实验以供比较。
- **灵活输入**: 支持自定义基准文件、跟踪文件和拓扑结构。
- **模块化结构**: 易于扩展以支持新的策略或应用。

## 硬件/软件要求

本框架在以下环境中开发和测试：
- CPU: 13th Gen Intel(R) Core(TM) i7-13650HX
- 内存: 32 GB DDR5
- 存储: 1TB NVMe SSD
- 操作系统: Linux (Ubuntu 22.04.5 LTS)
- 软件依赖: Python 3.10.12

## 如何开始使用

### 1. 克隆代码库
```bash
git clone https://github.com/FCAS-LAB/FHE_ESWEEK.git
cd FHE_ESWEEK
```

### 2. 运行 CKKS_Resnet50 演示（双层规划算法 + 消融实验）
```bash
bash run.sh
```

### 3. 运行单个基准测试
使用以下命令将执行本文中关于 CKKS_ResNet50、CKKS_CNN、CKKS_MLP、CKKS_VGG 的所有实验，包括详细的双层规划过程和相应的消融实验。

- **CKKS_ResNet50:**
```bash
python main.py \
  --time_file="./src/Resnet50/Resnet50_time.txt" \
  --bench_file="./src/Resnet50/Resnet50_bench.txt" \
  --trace_file="./src/Resnet50/Resnet50_trace.txt" \
  --topo_file="./src/Resnet50/Resnet50_gv.txt" \
  --show_file="./src/Resnet50/Resnet50_show.txt"
```

- **CKKS_CNN:**
```bash
python main.py \
  --time_file="./src/CNN/CNN_time.txt" \
  --bench_file="./src/CNN/CNN_bench.txt" \
  --trace_file="./src/CNN/CNN_trace.txt" \
  --topo_file="./src/CNN/CNN_gv.txt" \
  --show_file="./src/CNN/CNN_show.txt"
```

- **CKKS_MLP:**
```bash
python main.py \
  --time_file="./src/MLP/MLP_time.txt" \
  --bench_file="./src/MLP/MLP_bench.txt" \
  --trace_file="./src/MLP/MLP_trace.txt" \
  --topo_file="./src/MLP/MLP_gv.txt" \
  --show_file="./src/MLP/MLP_show.txt"
```

- **CKKS_VGG:**
```bash
python main.py \
  --time_file="./src/VGG/VGG_time.txt" \
  --bench_file="./src/VGG/VGG_bench.txt" \
  --trace_file="./src/VGG/VGG_trace.txt" \
  --topo_file="./src/VGG/VGG_gv.txt" \
  --show_file="./src/VGG/VGG_show.txt"
```

## 输出解释

### 双层优化报告
```
================ Running Bilevel Algorithm ================
# Iteration progress with time consumption updates
the original total time consumption: ... 
total time consumption after optimization iter 0: ...
total time consumption after optimization iter 1: ...
...
final optimized performance: ...  # Optimized execution time
final optimized topology (in the format of adjacency matrix):  # Optimized topology
1:[...]
2:[...]
3:[...]
...
90:[...]
```

### 消融实验
```
================ Ablation Experiments ================
Mapping Ablation:
time consumption of random_mapping + ring-ring topology: ...
time consumption of proposed mapping + ring-ring topology: ...
time consumption of random mapping + mesh-mesh topology: ...
time consumption of proposed mapping + mesh-mesh topology: ...
-----------------------------------------------------
Topology Ablation:
...
```