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

To run the program, follow the steps below for either **script execution** or **manual command execution**.

### Run via Script

```sh
git clone https://github.com/FCAS-LAB/FHE_ESWEEK.git
cd FHE_ESWEEK
bash run.sh
```

This script will run and display the optimization process for ResNet50.

### Run via Terminal Command

You can also run ResNet50 and other benchmarks (e.g., CNN) using command line.

- To run ResNet50:

```sh
git clone https://github.com/FCAS-LAB/FHE_ESWEEK.git
cd FHE_ESWEEK
python main.py \
  --time_file="./src/Resnet50/Resnet50_time.txt" \
  --bench_file="./src/Resnet50/Resnet50_bench.txt" \
  --trace_file="./src/Resnet50/Resnet50_trace.txt" \
  --topo_file="./src/Resnet50/Resnet50_gv.txt" \
  --show_file="./src/Resnet50/Resnet50_show.txt"
```

- To run CNN:

```sh
git clone https://github.com/FCAS-LAB/FHE_ESWEEK.git
cd FHE_ESWEEK
python main.py \
  --time_file="./src/CNN/CNN_time.txt" \
  --bench_file="./src/CNN/CNN_bench.txt" \
  --trace_file="./src/CNN/CNN_trace.txt" \
  --topo_file="./src/CNN/CNN_gv.txt" \
  --show_file="./src/CNN/CNN_show.txt"
```

- To run MLP:

```sh
git clone https://github.com/FCAS-LAB/FHE_ESWEEK.git
cd FHE_ESWEEK
python main.py \
  --time_file="./src/MLP/MLP_time.txt" \
  --bench_file="./src/MLP/MLP_bench.txt" \
  --trace_file="./src/MLP/MLP_trace.txt" \
  --topo_file="./src/MLP/MLP_gv.txt" \
  --show_file="./src/MLP/MLP_show.txt"
```

- To run VGG:

```sh
git clone https://github.com/FCAS-LAB/FHE_ESWEEK.git
cd FHE_ESWEEK
python main.py \
  --time_file="./src/VGG/VGG_time.txt" \
  --bench_file="./src/VGG/VGG_bench.txt" \
  --trace_file="./src/VGG/VGG_trace.txt" \
  --topo_file="./src/VGG/VGG_gv.txt" \
  --show_file="./src/VGG/VGG_show.txt"
```