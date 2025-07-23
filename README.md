# Homomorphic Encryption Task Graph Mapping and Inter- and Intra-Chiplet Topology Generation Framework

This repository provides a Python-based framework for Design Space Exploration (DSE) of homomorphic encryption applications in multi-chiplet systems. It includes bilevel optimization algorithms, mapping and topology generation ablation experiments, and related experiment scripts.

## Key Features

- **Bilevel Optimization Algorithm**: Optimize mapping and topology jointly.
- **Baseline Ablations**: Isolated experiments on mapping and topology for comparison.
- **Flexible Input**: Support for custom benchmark files, trace files, and topologies.
- **Modular Structure**: Easy to extend for new strategies or applications.

## Hardware/Software Requirements

This framework was developed and tested in the following environment:
- Processor: 13th Gen Intel(R) Core(TM) i7-13650HX
- RAM: 32 GB DDR5
- Storage: 1TB NVMe SSD
- Operating System: Linux (Ubuntu 22.04.5 LTS)
- Software Dependency: Python 3.10.12

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/FCAS-LAB/FHE_ESWEEK.git
cd FHE_ESWEEK
```

### 2. Run CKKS_Resnet50 Demo (Bilevel Optimization Algorithm + Ablation Experiments)
```bash
bash run.sh
```

### 3. Run Individual Benchmarks
Using the commands below will execute all the experiments regarding CKKS_ResNet50, CKKS_CNN, CKKS_MLP, CKKS_VGG in this paper, including detailed bilevel optimization process and corresponding ablation experiments.

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
  --time_file="./src/MLP/MLP_time.txt" \
  --bench_file="./src/MLP/MLP_bench.txt" \
  --trace_file="./src/MLP/MLP_trace.txt" \
  --topo_file="./src/MLP/MLP_gv.txt" \
  --show_file="./src/MLP/MLP_show.txt"
```

## Output Interpretation

### Bilevel Optimization Report
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

### Ablation Studies
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
